"""
finetune_loop.py -- CORRECTED, PARAMETERISED, and now CLI-invocable for
concurrent multi-process ablation runs (run_ablation_parallel.py launches
this as separate OS processes via subprocess, not multiprocessing --
GRF's C++ engine and torch are not reliably fork-safe, so genuinely
separate processes are the correct choice here, not a thread/fork pool).

torch.set_num_threads(1) is set at import time, unconditionally. Without
this, PyTorch tries to use ALL CPU cores for its own ops WITHIN each
process -- run N of these concurrently and every worker fights every
other worker for every core, which can make concurrent execution SLOWER
than sequential. This is not optional when running >1 instance at once.
"""

import torch
torch.set_num_threads(1)

import os, sys, time, copy
import numpy as np
import wandb

PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")

sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.join(GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "discriminator"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "policy"))

import gfootball.env as football_env
from minimal_state import MinimalState
from enhanced_LightActionMask_5 import FeatureEncoder
from record_baseline_rollout import pick_reference_agent

from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from simple_gae import compute_gae
from finetune_objective import (three_term_loss, compute_style_reward,
                                 AlignmentTaxScheduler, ALPHA, LAMBDA_KL_INIT)
from live_context_sampler import LiveAgentBuffer
from evaluate_policy import evaluate_policy

from discriminator_model import Discriminator
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from feature_derivation import compute_raw_features
from feature_normalization import FeatureNormaliser
from held_out_split import make_episode_split, split_features_by_episode
from build_expert_dataset import classify_bin

ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
CRITIC_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/critic.pt")
DISC_CKPT = os.path.join(PROJECT_ROOT, "src/discriminator/discriminator_phase_b_checkpoint.pt")
NORMALISER_PATH = os.path.join(PROJECT_ROOT, "src/discriminator/feature_normaliser.pkl")
EXPERT_CACHE = os.path.join(PROJECT_ROOT, "src/discriminator/expert_features_cache.npz")

MAX_STEPS = 3000
GAMMA = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 4
DISC_UPDATE_EVERY = 10


def build_env():
    return football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0, render=False)


def build_disc_input(o, sticky, action, normaliser):
    step = {
        'left_team': o['left_team'], 'left_team_direction': o['left_team_direction'],
        'right_team': o['right_team'], 'right_team_direction': o['right_team_direction'],
        'ball': o['ball'], 'ball_direction': o['ball_direction'],
        'action_captured': action, 'sticky_actions': sticky,
        'steps_left': o['steps_left'],
        'score_left': o['score'][0], 'score_right': o['score'][1],
    }
    return normaliser.transform(compute_raw_features(step))


def collect_rollout(actor, critic, encoder, env, normaliser, use_context=True):
    raw_obs = env.reset()
    obs_log, ctx_log, act_log, lp_log, val_log, mask_log = [], [], [], [], [], []
    disc_log, bins_log, rew_log, done_log = [], [], [], []

    step, done = 0, False
    while not done and step < MAX_STEPS:
        o = raw_obs[0]
        t_norm = o['steps_left'] / 3001.0
        dscore = int(o['score'][0]) - int(o['score'][1])
        ctx_vals = ([t_norm, np.clip(dscore, -3, 3) / 3.0] if use_context else [0.0, 0.0])
        ctx_t = torch.as_tensor(np.array(ctx_vals, dtype=np.float32)).unsqueeze(0).repeat(4, 1)

        obs_192, masks = [], []
        for i in range(4):
            s = MinimalState(n_player=5); s.set_obs(raw_obs[i])
            obs_192.append(encoder.encode_each(s))
            masks.append(encoder.get_available_actions(raw_obs[i], 0.0, []))
        obs_t = torch.as_tensor(np.stack(obs_192), dtype=torch.float32)
        mask_t = torch.as_tensor(np.stack(masks), dtype=torch.float32)

        with torch.no_grad():
            actions, lps, _, _ = actor(obs_t, ctx_t, mask_t, explore=True)
            vals = critic(obs_t, ctx_t)

        ref_i, _ = pick_reference_agent(raw_obs, range(4))
        disc_log.append(build_disc_input(o, raw_obs[ref_i]['sticky_actions'],
                                          int(actions[ref_i].item()), normaliser))
        bins_log.append(classify_bin(t_norm, dscore))

        obs_log.append(obs_t); ctx_log.append(ctx_t); act_log.append(actions)
        lp_log.append(lps); val_log.append(vals); mask_log.append(mask_t)

        raw_obs, reward, done, _ = env.step(actions.tolist())
        rew_log.append(float(np.sum(reward)))
        done_log.append(float(done))
        step += 1

    o = raw_obs[0]
    t_norm = o['steps_left'] / 3001.0
    dscore = int(o['score'][0]) - int(o['score'][1])
    ctx_vals = ([t_norm, np.clip(dscore, -3, 3) / 3.0] if use_context else [0.0, 0.0])
    ctx_t = torch.as_tensor(np.array(ctx_vals, dtype=np.float32)).unsqueeze(0).repeat(4, 1)
    obs_192 = []
    for i in range(4):
        s = MinimalState(n_player=5); s.set_obs(raw_obs[i])
        obs_192.append(encoder.encode_each(s))
    with torch.no_grad():
        boot = critic(torch.as_tensor(np.stack(obs_192), dtype=torch.float32), ctx_t)

    return {
        "obs": torch.cat(obs_log), "ctx": torch.cat(ctx_log),
        "actions": torch.cat(act_log), "old_log_probs": torch.cat(lp_log),
        "mask": torch.cat(mask_log), "values": torch.cat(val_log + [boot]),
        "disc_feat": np.stack(disc_log), "bins": np.array(bins_log),
        "rewards": torch.tensor(rew_log, dtype=torch.float32),
        "dones": torch.tensor(done_log, dtype=torch.float32),
        "final_score": (o['score'][0], o['score'][1]), "n_steps": step,
    }


def run(condition="MAGAIL-C", seed=0, max_iterations=100, use_kl=False,
        shuffle_context=False, use_context=True, lam_init=LAMBDA_KL_INIT,
        eval_every=30, eval_episodes=10, wandb_project="magail-c", frozen=False):

    torch.manual_seed(seed); np.random.seed(seed)
    run_name = f"{condition}_seed{seed}"
    print(f"\n{'='*70}\n{run_name}  ({max_iterations} iters, use_kl={use_kl}, "
          f"shuffle_context={shuffle_context}, use_context={use_context}, frozen={frozen})\n{'='*70}")

    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
    frozen_critic = torch.load(CRITIC_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor)
    critic = ContextConditionedCritic(frozen_critic)

    if frozen:
        # MAPPO-baseline: NO training, NO discriminator, NO PPO. context_net
        # is zero-init (verified elsewhere) so this actor's behaviour is
        # IDENTICAL to the raw checkpoint regardless -- just run a real,
        # properly-sized evaluation (locked minimum: >=50 episodes) and stop.
        actor.eval()
        wandb.init(project=wandb_project, name=run_name, reinit=True,
                  config={"condition": condition, "seed": seed, "frozen": True})
        encoder = FeatureEncoder()
        t0 = time.time()
        ev = evaluate_policy(actor, encoder, build_env,
                             n_episodes=max(eval_episodes, 50), max_steps=MAX_STEPS)
        elapsed = time.time() - t0
        print(f"{run_name}: win={ev['win_rate']:.3f} SAP={ev['sap_mean']:.2f}% "
              f"MECHA={ev['mecha_mean']:.4f} ({elapsed/60:.1f}min)")
        wandb.log({f"eval/{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
        wandb.finish()
        result = {"condition": condition, "seed": seed, "iterations": 0,
                  "halted": False, "wall_time_sec": elapsed, **ev}
        import json
        with open(f"result_{run_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    actor_star = copy.deepcopy(actor)
    for p in actor_star.parameters():
        p.requires_grad_(False)
    actor_star.eval()

    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(DISC_CKPT, map_location="cpu")["model_state_dict"])
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-5, weight_decay=1e-4)

    normaliser = FeatureNormaliser(); normaliser.load(NORMALISER_PATH)
    encoder = FeatureEncoder()

    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr_ep, ho_ep = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), _ = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr_ep, ho_ep)
    expert_sampler = BalancedContextSampler(train_feat, train_bins, name="expert_train")
    target_props = sqrt_scaled_target(expert_sampler.cell_counts)

    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING['t_norm_max'],
                                   delta_score_sign=LATE_WINNING['delta_score_sign'])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING['t_norm_max'],
                                   delta_score_sign=LATE_LOSING['delta_score_sign'])
    disc_trainer = DiscriminatorTrainer(
        discriminator, disc_opt, expert_sampler=expert_sampler, eta=0.3, gamma_swap=6.0,
        swap_lw_features=train_feat[m_lw], swap_ll_features=train_feat[m_ll])

    policy_opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    scheduler = AlignmentTaxScheduler(lambda_init=lam_init)
    live_buffer = LiveAgentBuffer(max_episodes=20)
    rng = np.random.default_rng(seed)

    wandb.init(project=wandb_project, name=run_name, reinit=True, config={
        "condition": condition, "seed": seed, "max_iterations": max_iterations,
        "use_kl": use_kl, "shuffle_context": shuffle_context, "use_context": use_context,
        "alpha": ALPHA, "lam_init": lam_init, "eval_every": eval_every,
        "eval_episodes": eval_episodes,
    })

    env = build_env()
    halted, iter_times = False, []
    t_run_start = time.time()

    for it in range(1, max_iterations + 1):
        t0 = time.time()
        batch = collect_rollout(actor, critic, encoder, env, normaliser, use_context=use_context)

        r_style = compute_style_reward(discriminator, torch.as_tensor(batch["disc_feat"], dtype=torch.float32))

        # FIX A: style reward enters the RETURN, not the loss.
        combined_reward = batch["rewards"] + ALPHA * r_style

        T = batch["n_steps"]
        values = batch["values"].view(T + 1, 4)
        rewards = combined_reward.unsqueeze(1).repeat(1, 4)
        dones = batch["dones"].unsqueeze(1).repeat(1, 4)
        advs = torch.stack([compute_gae(rewards[:, a], values[:, a], dones[:, a],
                                         gamma=GAMMA, gae_lambda=GAE_LAMBDA)[0]
                            for a in range(4)], dim=1).flatten()

        loss_diag = {}
        for _ in range(K_EPOCHS):
            _, new_lp, _, dist_theta = actor(batch["obs"], batch["ctx"], batch["mask"],
                                              explore=False, actions=batch["actions"])
            with torch.no_grad():
                _, _, _, dist_star = actor_star(batch["obs"], torch.zeros_like(batch["ctx"]),
                                                 batch["mask"], explore=False, actions=batch["actions"])
            loss, loss_diag = three_term_loss(new_lp, batch["old_log_probs"], advs,
                                               dist_theta, dist_star,
                                               lam=scheduler.lam, use_kl=use_kl)
            policy_opt.zero_grad(); loss.backward(); policy_opt.step()

        live_buffer.add_episode(batch["disc_feat"], batch["bins"])

        disc_metrics = {}
        disc_skipped = 0.0
        if it % DISC_UPDATE_EVERY == 0:
            d_acc = disc_trainer.get_recent_accuracy()
            if d_acc is None or d_acc < 0.97:
                agent_sampler = live_buffer.to_sampler()
                if agent_sampler is not None:
                    (ef, eb), (af, ab) = balanced_batch(expert_sampler, agent_sampler, batch_size=128,
                                                         target_props=target_props,
                                                         rng=np.random.default_rng(it + seed * 1000),
                                                         on_empty='skip')
                    if shuffle_context:
                        ef = ef.copy(); af = af.copy()
                        ef[:, -2:] = ef[rng.permutation(len(ef)), -2:]
                        af[:, -2:] = af[rng.permutation(len(af)), -2:]
                    disc_metrics = disc_trainer.step(ef, af, expert_cells=eb, agent_cells=ab)
            else:
                disc_skipped = 1.0

        train_elapsed = time.time() - t0
        iter_times.append(train_elapsed)

        log = {
            "iteration": it, "episode_steps": T,
            "episode_sap": 100.0 * float(batch["disc_feat"][:, 135].mean()),
            "r_style_mean": r_style.mean().item(),
            "task_reward_sum": batch["rewards"].sum().item(),
            "disc_update_skipped": disc_skipped,
            **{f"loss/{k}": v for k, v in loss_diag.items() if isinstance(v, (int, float))},
            **{f"disc/{k}": v for k, v in disc_metrics.items() if isinstance(v, (int, float))},
        }

        if it % eval_every == 0:
            t_e = time.time()
            ev = evaluate_policy(actor, encoder, build_env,
                                 n_episodes=eval_episodes, max_steps=MAX_STEPS)
            new_lam, halt_wr, reason = scheduler.update(ev["win_rate"])
            health = disc_trainer.health()
            log.update({f"eval/{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
            log["disc_health"] = health
            print(f"\n[{run_name} it{it}] EVAL ({time.time()-t_e:.0f}s): win={ev['win_rate']:.3f} "
                  f"SAP={ev['sap_mean']:.2f}% MECHA={ev['mecha_mean']:.4f} "
                  f"CSI={ev['csi_sap_proxy']} health={health} lam={new_lam:.4f}")
            # r_style = logit(D) is unbounded and keeps varying at high accuracy,
            # so the vanishing-gradient rationale for halting does not apply here.
            if health == "saturating":
                log["disc_saturating"] = 1.0
                
            if halt_wr:
                print(f"  KILL-SWITCH (win rate): {reason}"); halted = True

        wandb.log(log)

        if it % 10 == 0 or halted:
            mean_it = np.mean(iter_times[-20:])
            remaining = (max_iterations - it) * mean_it
            print(f"[{run_name} it{it}/{max_iterations}] {train_elapsed:.1f}s  "
                  f"score={batch['final_score']}  r_style={r_style.mean():.3f}  "
                  f"ETA {remaining/60:.1f}min")

        if halted:
            break

    ckpt = f"ablation_{run_name}.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "discriminator": discriminator.state_dict(), "condition": condition,
                "seed": seed, "iterations_completed": it, "halted": halted}, ckpt)
    env.close(); wandb.finish()
    total = time.time() - t_run_start
    result = {"condition": condition, "seed": seed, "iterations": it,
              "halted": halted, "wall_time_sec": total}
    print(f"{run_name} done in {total/60:.1f}min, {it} iters, halted={halted} -> {ckpt}")

    # Written per-run, own filename -- concurrent processes CANNOT safely
    # share-write one progress file (last writer wins, others lost). The
    # launcher (run_ablation_parallel.py) reads these individually rather
    # than relying on any single shared file.
    import json
    with open(f"result_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-iterations", type=int, required=True)
    p.add_argument("--use-kl", type=int, default=0)          # 0/1, not store_true --
                                                        # explicit from subprocess call
    p.add_argument("--shuffle-context", type=int, default=0)
    p.add_argument("--use-context", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--lam-init", type=float, default=None)
    p.add_argument("--frozen", type=int, default=0)   # 1 = MAPPO-baseline: eval only, no training
    args = p.parse_args()

    run(condition=args.condition, seed=args.seed, max_iterations=args.max_iterations,
        use_kl=bool(args.use_kl), shuffle_context=bool(args.shuffle_context),
        use_context=bool(args.use_context), eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        lam_init=(args.lam_init if args.lam_init is not None else LAMBDA_KL_INIT),
        frozen=bool(args.frozen))
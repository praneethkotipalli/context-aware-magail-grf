"""
finetune_loop.py -- STAGE 1 FIXED, v2.

Changes vs the version this replaces:

  [B1/R4] Single blended r_style -> two separately-weighted terms,
          combined = task + ALPHA_MARG*r_marg + ALPHA_INT*r_int, one
          shared GAE pass over the sum.

  [B2/B3] SUPERSEDES the earlier runtime-context-shuffle design. That
          design fed a SINGLE frozen D_int a manipulated input at
          inference time -- which does not answer the causal question
          SC/NC exist to ask (whether ACCESS TO CONTEXT DURING TRAINING
          caused the discriminator to learn something real). Manipulating
          the input of a model already trained on true context tests
          robustness-to-corruption, a different question.

          Fixed here: three GENUINELY SEPARATE, separately-trained D_int
          checkpoints (see train_D_int_variant.py), selected by
          condition. D_int is FROZEN either way (no online training,
          no optimizer) -- what changes is WHICH checkpoint is loaded,
          not what it's shown at runtime. All three always receive the
          real, unmodified context; the ablation lives entirely in what
          each checkpoint learned (or, for NC, could ever learn) during
          its own separate training run.

            true    -> D_int_C_production.pt   (TinyInteract, true context)
            zero    -> D_int_NC_production.pt  (SprintOnlyDiscriminator --
                       architecturally has no context input at all)
            shuffle -> D_int_SC_production.pt  (TinyInteract, trained on
                       a corpus with context permuted once, data-level,
                       before training -- see build_shuffled_context_caches.py)

  [B4]    Already fixed at the constant-definition sites
          (context_shift_scoring.py / metrics.py).

  [B6]    eval_every 30->50, eval_episodes 10->30; two-stage kill-switch.

  Also: per-episode style-reward centering replaced with a RUNNING mean
  of the RAW logit, not the already-centered output.
"""

import torch
torch.set_num_threads(1)

import os, sys, time, copy, json
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
# [B1/R4] ALPHA (single constant) and compute_style_reward no longer used --
# replaced by ALPHA_MARG/ALPHA_INT below and the local _style_reward() helper.
from finetune_objective import three_term_loss, AlignmentTaxScheduler, LAMBDA_KL_INIT
from live_context_sampler import LiveAgentBuffer
from evaluate_policy import evaluate_policy

from discriminator_model import Discriminator
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING  # [B4] already 0.2222 at source
from feature_derivation import compute_raw_features
from feature_normalization import FeatureNormaliser
from held_out_split import make_episode_split, split_features_by_episode
from build_expert_dataset import classify_bin
from counterfactual_gate import run_counterfactual_gate

# [B2/B3] v2: three separate D_int architectures/checkpoints, no runtime
# context manipulation, disc_context.py no longer used/needed.
from discriminator_candidates import TinyInteract, SprintOnlyDiscriminator

ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
CRITIC_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/critic.pt")
DISC_CKPT = os.path.join(PROJECT_ROOT, "src/discriminator/discriminator_phase_b_checkpoint.pt")
NORMALISER_PATH = os.path.join(PROJECT_ROOT, "src/discriminator/feature_normaliser.pkl")
EXPERT_CACHE = os.path.join(PROJECT_ROOT, "src/discriminator/expert_features_cache.npz")

# [B2/B3] one checkpoint + architecture per condition, produced by
# train_D_int_variant.py --variant {C,NC,SC}. Keyed on the SAME mode
# strings the old backward-compat bridge used, so CLI callers don't change.
D_INT_CKPTS = {
    "true":    os.path.join(PROJECT_ROOT, "src/discriminator/D_int_C_production.pt"),
    "zero":    os.path.join(PROJECT_ROOT, "src/discriminator/D_int_NC_production.pt"),
    "shuffle": os.path.join(PROJECT_ROOT, "src/discriminator/D_int_SC_production.pt"),
}
D_INT_MODELS = {
    "true":    TinyInteract,
    "zero":    SprintOnlyDiscriminator,   # architecturally no context input at all
    "shuffle": TinyInteract,
}

MAX_STEPS = 3000
GAMMA = 0.99
GAE_LAMBDA = 0.95
K_EPOCHS = 4
DISC_UPDATE_EVERY = 10
STYLE_CLIP = 5.0
STYLE_EMA = 0.99

# [B1/R4] PROVISIONAL starting values from measure_alpha.py's raw-sum-parity
# suggestion. NOT yet confirmed by the alpha-bracketing pilot -- treat as
# the conservative floor of a small search. Override with --alpha-marg /
# --alpha-int once the pilot completes.
ALPHA_MARG = 0.0048
ALPHA_INT = 0.0026


def _resolve_disc_int_mode(use_context, shuffle_context):
    """Backward-compat bridge: run_ablation_parallel.py's existing
    CORE_CONDITIONS list constructs jobs via the OLD --use-context /
    --shuffle-context flags. Map those combinations onto which D_int
    checkpoint/architecture to load, so the launcher's condition
    definitions (MAGAIL-C: True/False, MAGAIL-SC: True/True,
    MAGAIL-NC: False/False) keep working unmodified."""
    if shuffle_context:
        return 'shuffle'
    if not use_context:
        return 'zero'
    return 'true'


def build_env():
    return football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0, render=False)


def build_disc_input(o, sticky, action, normaliser):
    """UNCHANGED. Always builds TRUE context. This feeds D_marg (always
    true-context, identical across every condition) AND, as of v2, also
    D_int directly -- since D_int's ablation now lives entirely in WHICH
    checkpoint is loaded, not in what its input looks like. Every
    discriminator in this file receives the real, unmodified context."""
    step = {
        'left_team': o['left_team'], 'left_team_direction': o['left_team_direction'],
        'right_team': o['right_team'], 'right_team_direction': o['right_team_direction'],
        'ball': o['ball'], 'ball_direction': o['ball_direction'],
        'action_captured': action, 'sticky_actions': sticky,
        'steps_left': o['steps_left'],
        'score_left': o['score'][0], 'score_right': o['score'][1],
    }
    return normaliser.transform(compute_raw_features(step))


def _style_reward(discriminator, feats, running_mean, clip=STYLE_CLIP, ema=STYLE_EMA):
    """Centers on a RUNNING mean of the RAW logit, not the per-batch mean.
    Per-episode centering destroys the between-episode component of the
    signal. Returns (centered_reward_tensor, updated_running_mean_float).
    running_mean=None on the first call -> initializes from this batch."""
    with torch.no_grad():
        raw = discriminator(feats).clamp(-clip, clip)
    raw_mean = raw.mean().item()
    center = raw_mean if running_mean is None else running_mean
    reward = raw - center
    new_running_mean = raw_mean if running_mean is None else (ema * running_mean + (1 - ema) * raw_mean)
    return reward, new_running_mean


def collect_rollout(actor, critic, encoder, env, normaliser):
    """[B2] use_context parameter REMOVED. The policy is ALWAYS
    context-conditioned, every condition."""
    raw_obs = env.reset()
    obs_log, ctx_log, act_log, lp_log, val_log, mask_log = [], [], [], [], [], []
    disc_log, bins_log, rew_log, done_log = [], [], [], []

    step, done = 0, False
    while not done and step < MAX_STEPS:
        o = raw_obs[0]
        t_norm = o['steps_left'] / 3001.0
        dscore = int(o['score'][0]) - int(o['score'][1])
        ctx_vals = [t_norm, np.clip(dscore, -3, 3) / 3.0]   # [B2] always real
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
    ctx_vals = [t_norm, np.clip(dscore, -3, 3) / 3.0]   # [B2] always real
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
        eval_every=50, eval_episodes=30, wandb_project="magail-c-v2", frozen=False,
        disc_int_context=None, alpha_marg=None, alpha_int=None, no_anneal=False):

    torch.manual_seed(seed); np.random.seed(seed)
    run_name = f"{condition}_seed{seed}"

    # [B2/B3] resolve WHICH D_int checkpoint/architecture this run uses.
    # This now picks a CHECKPOINT, not an input manipulation -- both D_marg
    # and D_int always see the real, unmodified context at runtime.
    if disc_int_context is None:
        disc_int_context = _resolve_disc_int_mode(use_context, shuffle_context)
    d_int_ckpt_path = D_INT_CKPTS[disc_int_context]
    d_int_model_cls = D_INT_MODELS[disc_int_context]

    ALPHA_M = ALPHA_MARG if alpha_marg is None else alpha_marg
    ALPHA_I = ALPHA_INT if alpha_int is None else alpha_int

    print(f"\n{'='*70}\n{run_name}  ({max_iterations} iters, use_kl={use_kl}, "
          f"disc_int_condition={disc_int_context} ({d_int_model_cls.__name__} <- "
          f"{os.path.basename(d_int_ckpt_path)}), "
          f"alpha_marg={ALPHA_M}, alpha_int={ALPHA_I}, frozen={frozen})\n{'='*70}")

    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
    frozen_critic = torch.load(CRITIC_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor)
    critic = ContextConditionedCritic(frozen_critic)

    if frozen:
        # MAPPO-baseline: unaffected by any Stage 1 fix. Independent of
        # everything else here; can be launched anytime, in parallel.
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
        with open(f"result_{run_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    actor_star = copy.deepcopy(actor)
    for p in actor_star.parameters():
        p.requires_grad_(False)
    actor_star.eval()

    # --- D_marg: ALWAYS true context, identical across every condition,
    #     trains online exactly as before (unaffected by B2/B3) ---
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(DISC_CKPT, map_location="cpu")["model_state_dict"])
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-5, weight_decay=1e-4)

    # --- D_int: FROZEN, loaded once, no optimizer, no trainer. WHICH
    #     checkpoint/architecture is loaded encodes the condition; its
    #     INPUT is always the real, unmodified context. [B2/B3 v2] ---
    D_int = d_int_model_cls()
    D_int.load_state_dict(torch.load(d_int_ckpt_path, map_location="cpu")["model_state_dict"])
    D_int.eval()
    for p in D_int.parameters():
        p.requires_grad_(False)

    normaliser = FeatureNormaliser(); normaliser.load(NORMALISER_PATH)
    encoder = FeatureEncoder()

    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr_ep, ho_ep = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr_ep, ho_ep)
    expert_sampler = BalancedContextSampler(train_feat, train_bins, name="expert_train")
    target_props = sqrt_scaled_target(expert_sampler.cell_counts)

    # [B4] canonical threshold at source. D_marg's swap population, unaffected by condition.
    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING['t_norm_max'],
                                   delta_score_sign=LATE_WINNING['delta_score_sign'])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING['t_norm_max'],
                                   delta_score_sign=LATE_LOSING['delta_score_sign'])
    disc_trainer = DiscriminatorTrainer(
        discriminator, disc_opt, expert_sampler=expert_sampler, eta=0.3, gamma_swap=6.0,
        swap_lw_features=train_feat[m_lw], swap_ll_features=train_feat[m_ll])

    policy_opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    scheduler = AlignmentTaxScheduler(lambda_init=lam_init, anneal_patience=(float('inf') if no_anneal else 5))    
    live_buffer = LiveAgentBuffer(max_episodes=20)
    rng = np.random.default_rng(seed)

    style_rm_marg, style_rm_int = None, None

    wandb.init(project=wandb_project, name=run_name, reinit=True, config={
        "condition": condition, "seed": seed, "max_iterations": max_iterations,
        "use_kl": use_kl, "disc_int_condition": disc_int_context,
        "d_int_architecture": d_int_model_cls.__name__, "d_int_ckpt": d_int_ckpt_path,
        "alpha_marg": ALPHA_M, "alpha_int": ALPHA_I,
        "lam_init": lam_init, "eval_every": eval_every, "eval_episodes": eval_episodes,
        "raw_use_context_flag": use_context, "raw_shuffle_context_flag": shuffle_context,
    })

    env = build_env()
    halted, iter_times = False, []
    t_run_start = time.time()

    for it in range(1, max_iterations + 1):
        t0 = time.time()
        batch = collect_rollout(actor, critic, encoder, env, normaliser)  # [B2] no use_context arg

        # --- [B1/R4 + B2/B3 v2] dual style reward: SAME true-context array
        #     feeds BOTH discriminators. Only which D_int is loaded differs
        #     by condition -- there is no more runtime input manipulation. ---
        disc_feat = torch.as_tensor(batch["disc_feat"], dtype=torch.float32)

        r_marg, style_rm_marg = _style_reward(discriminator, disc_feat, style_rm_marg)
        r_int, style_rm_int = _style_reward(D_int, disc_feat, style_rm_int)

        # FIX A (pre-existing, unchanged): style enters the RETURN, not the loss.
        combined_reward = batch["rewards"] + ALPHA_M * r_marg + ALPHA_I * r_int

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

        live_buffer.add_episode(batch["disc_feat"], batch["bins"])  # true-context, D_marg's own buffer

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
                    # [B3] no shuffle branch here -- D_marg's BCE training is
                    # ALWAYS true-context, every condition. SC's effect lives
                    # entirely in which D_int checkpoint was loaded above.
                    disc_metrics = disc_trainer.step(ef, af, expert_cells=eb, agent_cells=ab)
            else:
                disc_skipped = 1.0

        train_elapsed = time.time() - t0
        iter_times.append(train_elapsed)

        log = {
            "iteration": it, "episode_steps": T,
            "episode_sap": 100.0 * float(batch["disc_feat"][:, 135].mean()),
            "r_marg_mean": r_marg.mean().item(), "r_int_mean": r_int.mean().item(),
            "style_rm_marg": style_rm_marg, "style_rm_int": style_rm_int,
            "task_reward_sum": batch["rewards"].sum().item(),
            "disc_update_skipped": disc_skipped,
            **{f"loss/{k}": v for k, v in loss_diag.items() if isinstance(v, (int, float))},
            **{f"disc/{k}": v for k, v in disc_metrics.items() if isinstance(v, (int, float))},
        }

        if it % eval_every == 0:
            t_e = time.time()
            ev = evaluate_policy(actor, encoder, build_env, n_episodes=eval_episodes, max_steps=MAX_STEPS)
            win_rate = ev["win_rate"]

            # [B6] two-stage kill-switch
            provisional_drop = (scheduler.baseline_win_rate - win_rate) / max(scheduler.baseline_win_rate, 1e-9)
            if provisional_drop > scheduler.kill_switch_drop:
                print(f"  [scheduler] {eval_episodes}-ep eval tripped kill-switch "
                      f"(win={win_rate:.3f}) -- re-checking at 100 episodes before halting")
                ev = evaluate_policy(actor, encoder, build_env, n_episodes=100, max_steps=MAX_STEPS)
                win_rate = ev["win_rate"]
                print(f"  [scheduler] 100-ep confirm: win={win_rate:.3f}")

            new_lam, halt_wr, reason = scheduler.update(win_rate)
            health = disc_trainer.health()
            log.update({f"eval/{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
            log["disc_health"] = health
            _, gate_summary = run_counterfactual_gate(discriminator, held_feat)  # D_marg's own gate
            for r in gate_summary["directions"]:
                log[f"gate/{r['direction']}_shift"] = r["mean_abs_shift"]
                log[f"gate/{r['direction']}_frac"] = r["frac_exceeding_0.1"]

            print(f"\n[{run_name} it{it}] EVAL ({time.time()-t_e:.0f}s): win={win_rate:.3f} "
                  f"SAP={ev['sap_mean']:.2f}% MECHA={ev['mecha_mean']:.4f} "
                  f"CSI={ev['csi_sap_proxy']} health={health} lam={new_lam:.4f}")
            if health == "saturating":
                log["disc_saturating"] = 1.0

            if halt_wr:
                print(f"  KILL-SWITCH (win rate): {reason}"); halted = True

        wandb.log(log)

        if it % 10 == 0 or halted:
            mean_it = np.mean(iter_times[-20:])
            remaining = (max_iterations - it) * mean_it
            print(f"[{run_name} it{it}/{max_iterations}] {train_elapsed:.1f}s  "
                  f"score={batch['final_score']}  r_marg={r_marg.mean():.3f}  r_int={r_int.mean():.3f}  "
                  f"ETA {remaining/60:.1f}min")

        if halted:
            break

    ckpt = f"ablation_{run_name}.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "discriminator": discriminator.state_dict(),   # D_marg only -- D_int is fixed, path recorded below
                "d_int_ckpt_path": d_int_ckpt_path, "d_int_architecture": d_int_model_cls.__name__,
                "condition": condition, "seed": seed, "iterations_completed": it, "halted": halted,
                "disc_int_condition": disc_int_context,
                "alpha_marg": ALPHA_M, "alpha_int": ALPHA_I}, ckpt)
    env.close(); wandb.finish()
    total = time.time() - t_run_start
    result = {"condition": condition, "seed": seed, "iterations": it,
              "halted": halted, "wall_time_sec": total,
              "disc_int_condition": disc_int_context, "d_int_ckpt": d_int_ckpt_path,
              "alpha_marg": ALPHA_M, "alpha_int": ALPHA_I}
    print(f"{run_name} done in {total/60:.1f}min, {it} iters, halted={halted} -> {ckpt}")

    with open(f"result_{run_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-iterations", type=int, required=True)
    p.add_argument("--use-kl", type=int, default=0)
    # kept for backward compat with run_ablation_parallel.py's existing
    # CORE_CONDITIONS -- now only select WHICH D_int checkpoint to load.
    p.add_argument("--shuffle-context", type=int, default=0)
    p.add_argument("--use-context", type=int, default=1)
    p.add_argument("--disc-int-context", choices=["true", "zero", "shuffle"], default=None)
    p.add_argument("--eval-every", type=int, default=50)      # [B6] was 30
    p.add_argument("--eval-episodes", type=int, default=30)   # [B6] was 10
    p.add_argument("--lam-init", type=float, default=None)
    p.add_argument("--frozen", type=int, default=0)
    p.add_argument("--alpha-marg", type=float, default=None)
    p.add_argument("--alpha-int", type=float, default=None)
    p.add_argument("--no-anneal", type=int, default=0) # ADD THIS
    args = p.parse_args()

    run(condition=args.condition, seed=args.seed, max_iterations=args.max_iterations,
        use_kl=bool(args.use_kl), shuffle_context=bool(args.shuffle_context),
        use_context=bool(args.use_context), disc_int_context=args.disc_int_context,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        lam_init=(args.lam_init if args.lam_init is not None else LAMBDA_KL_INIT),
        frozen=bool(args.frozen), alpha_marg=args.alpha_marg, alpha_int=args.alpha_int,no_anneal=bool(args.no_anneal))
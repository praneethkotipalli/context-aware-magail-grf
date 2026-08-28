"""
pilot_finetune_iteration.py

ONE rollout episode + ONE PPO update, timed end-to-end. Purpose: a real
floor number for one iteration's cost, not a trained policy.

Explicitly NOT fixed today, by design:
  - discriminator is the FiLM checkpoint that FAILED the counterfactual
    gate. r_style here is a stand-in for timing the pipeline, not a
    validated behavioral-realism signal.
  - alpha=0.1 is untuned (see finetune_objective.py).
  - single episode, single worker, no parallelism -- real training
    throughput requires parallel rollout collection, not measured here.
    This number is a FLOOR estimate of per-iteration cost.
  - TWO separate feature encodings computed every step, deliberately:
    the policy's 192-dim light_malib encoding for actor/critic, and the
    discriminator's own 137-dim encoding for r_style. Not interchangeable,
    not a bug -- a real, permanent fact of this integration.
"""

import os, sys, time, copy
import numpy as np
import torch

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
from record_baseline_rollout import pick_reference_agent   # SAME convention the discriminator was trained on

from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from simple_gae import compute_gae
from finetune_objective import three_term_loss, compute_style_reward

from discriminator_model import Discriminator
from feature_derivation import compute_raw_features
from feature_normalization import normalize_features

ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
CRITIC_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/critic.pt")
DISCRIMINATOR_CHECKPOINT = os.path.join(PROJECT_ROOT, "src/discriminator/discriminator_phase_b_checkpoint.pt")

MAX_STEPS = 3000
GAMMA = 0.99
GAE_LAMBDA = 0.95


def build_context(steps_left, score_left, score_right):
    """Same raw->normalized convention as context_swap.py: t_norm raw
    [0,1], delta_score clip(-3,3)/3 -> [-1,1]."""
    t_norm = steps_left / 3001.0
    delta_score = np.clip(int(score_left) - int(score_right), -3, 3) / 3.0
    return np.array([t_norm, delta_score], dtype=np.float32)


def build_discriminator_input(raw_obs_shared, action_captured):
    step = {
        'left_team': raw_obs_shared['left_team'], 'left_team_direction': raw_obs_shared['left_team_direction'],
        'right_team': raw_obs_shared['right_team'], 'right_team_direction': raw_obs_shared['right_team_direction'],
        'ball': raw_obs_shared['ball'], 'ball_direction': raw_obs_shared['ball_direction'],
        'action_captured': action_captured,
        'steps_left': raw_obs_shared['steps_left'],
        'score_left': raw_obs_shared['score'][0], 'score_right': raw_obs_shared['score'][1],
    }
    return normalize_features(compute_raw_features(step))


def run():
    print("Loading models...")
    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu"); frozen_actor.eval()
    frozen_critic = torch.load(CRITIC_PATH, map_location="cpu"); frozen_critic.eval()

    actor = ContextConditionedActor(frozen_actor)
    critic = ContextConditionedCritic(frozen_critic)
    actor_star = copy.deepcopy(actor)
    for p in actor_star.parameters():
        p.requires_grad_(False)
    actor_star.eval()

    discriminator = Discriminator()
    ckpt = torch.load(DISCRIMINATOR_CHECKPOINT, map_location="cpu")
    discriminator.load_state_dict(ckpt["model_state_dict"])
    discriminator.eval()
    print(f"  discriminator: FiLM Phase B, held-out acc {ckpt.get('final_held_out_acc', 'n/a')} "
          f"-- GATE-FAILED, pilot/timing use only")

    optimizer = torch.optim.Adam(
        list(actor.context_net.parameters()) + list(critic.context_net.parameters()), lr=3e-4,
    )
    # NOTE: only context_net params trained today -- base/out stay frozen.
    # Whether to unfreeze base/out for real fine-tuning is a decision for
    # a later session, not today's pilot.

    encoder = FeatureEncoder()
    env = football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4, number_of_right_players_agent_controls=0,
        render=False,
    )

    print("\nStarting timed pilot iteration...")
    t_start = time.time()
    raw_obs = env.reset()

    obs_192_log, ctx_log, action_log, logprob_log, value_log, mask_log = [], [], [], [], [], []
    disc_feat_log, reward_log, done_log = [], [], []

    step = 0
    done = False
    t_rollout_start = time.time()
    while not done and step < MAX_STEPS:
        o_shared = raw_obs[0]
        ctx = build_context(o_shared['steps_left'], o_shared['score'][0], o_shared['score'][1])
        ctx_t = torch.as_tensor(ctx).unsqueeze(0).repeat(4, 1)

        obs_192, avail_masks = [], []
        for i in range(4):
            state = MinimalState(n_player=5)
            state.set_obs(raw_obs[i])
            obs_192.append(encoder.encode_each(state))
            avail_masks.append(encoder.get_available_actions(raw_obs[i], 0.0, []))
        obs_192_t = torch.as_tensor(np.stack(obs_192), dtype=torch.float32)
        mask_t = torch.as_tensor(np.stack(avail_masks), dtype=torch.float32)

        with torch.no_grad():
            actions, log_probs, _, _ = actor(obs_192_t, ctx_t, mask_t, explore=True)
            values = critic(obs_192_t, ctx_t)

        # SAME reference-player convention the discriminator was trained on
        _, ref_player_idx = pick_reference_agent(raw_obs, range(4))
        ref_agent_idx = ref_player_idx - 1
        action_captured_ref = int(actions[ref_agent_idx].item())
        disc_feat = build_discriminator_input(o_shared, action_captured_ref)

        obs_192_log.append(obs_192_t); ctx_log.append(ctx_t); action_log.append(actions)
        logprob_log.append(log_probs); value_log.append(values); mask_log.append(mask_t)
        disc_feat_log.append(disc_feat)

        raw_obs, reward, done, info = env.step(actions.tolist())
        reward_log.append(float(np.sum(reward)))  # shared team reward, methodology 3.1
        done_log.append(float(done))
        step += 1

    t_rollout_end = time.time()
    print(f"  rollout: {step} steps in {t_rollout_end - t_rollout_start:.2f}s "
          f"({step / (t_rollout_end - t_rollout_start):.1f} steps/sec)")

    # bootstrap value
    o_shared = raw_obs[0]
    ctx = build_context(o_shared['steps_left'], o_shared['score'][0], o_shared['score'][1])
    ctx_t = torch.as_tensor(ctx).unsqueeze(0).repeat(4, 1)
    obs_192 = [encoder.encode_each(MinimalState(n_player=5).__class__() and None or None) for _ in range(0)]  # placeholder removed below
    obs_192 = []
    for i in range(4):
        state = MinimalState(n_player=5); state.set_obs(raw_obs[i])
        obs_192.append(encoder.encode_each(state))
    obs_192_t = torch.as_tensor(np.stack(obs_192), dtype=torch.float32)
    with torch.no_grad():
        bootstrap_value = critic(obs_192_t, ctx_t)

    t_disc_start = time.time()
    disc_feat_batch = torch.as_tensor(np.stack(disc_feat_log), dtype=torch.float32)
    r_style = compute_style_reward(discriminator, disc_feat_batch)
    t_disc_end = time.time()
    print(f"  discriminator scoring: {len(disc_feat_log)} steps in {t_disc_end - t_disc_start:.3f}s")

    t_update_start = time.time()
    values_flat = torch.cat(value_log + [bootstrap_value]).view(step + 1, 4)
    rewards_flat = torch.tensor(reward_log).unsqueeze(1).repeat(1, 4)
    dones_flat = torch.tensor(done_log).unsqueeze(1).repeat(1, 4)

    all_advantages = []
    for agent_i in range(4):
        adv, _ = compute_gae(rewards_flat[:, agent_i], values_flat[:, agent_i], dones_flat[:, agent_i],
                              gamma=GAMMA, gae_lambda=GAE_LAMBDA)
        all_advantages.append(adv)
    advantages = torch.stack(all_advantages, dim=1).flatten()
    old_log_probs = torch.cat(logprob_log)

    obs_192_all = torch.cat(obs_192_log)
    ctx_all = torch.cat(ctx_log)
    actions_all = torch.cat(action_log)
    mask_all = torch.cat(mask_log)   # real masks, not a placeholder

    _, new_log_probs, _, dist_theta = actor(obs_192_all, ctx_all, mask_all, explore=False, actions=actions_all)
    with torch.no_grad():
        _, _, _, dist_star = actor_star(obs_192_all, torch.zeros_like(ctx_all), mask_all, explore=False, actions=actions_all)

    loss, diagnostics = three_term_loss(new_log_probs, old_log_probs, advantages, r_style, dist_theta, dist_star)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t_update_end = time.time()
    t_end = time.time()

    print(f"  PPO update: {t_update_end - t_update_start:.3f}s")
    print(f"\n  loss diagnostics: {diagnostics}")
    print(f"\n  TOTAL iteration time: {t_end - t_start:.2f}s")
    print(f"    rollout:      {t_rollout_end - t_rollout_start:.2f}s")
    print(f"    disc scoring: {t_disc_end - t_disc_start:.2f}s")
    print(f"    PPO update:   {t_update_end - t_update_start:.2f}s")
    print(f"\n  Single-worker, single-episode FLOOR estimate -- real training")
    print(f"  throughput requires parallel rollout collection, not measured here.")

    env.close()


if __name__ == '__main__':
    run()
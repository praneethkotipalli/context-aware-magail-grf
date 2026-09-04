"""
measure_alpha.py

Measures the actual magnitude gap between the style reward and the task
reward, then derives alpha. Replaces the unmeasured ALPHA=0.1 placeholder
in finetune_objective.py.

NO TRAINING. One rollout with the frozen baseline (not a fine-tuned policy)
scored by the GATE-PASSING discriminator. Frozen baseline deliberately:
alpha must be calibrated at the point fine-tuning STARTS, which is exactly
the frozen policy's behaviour distribution.

Reports both raw magnitudes and the derived alpha, per the requirement that
the measured ratio and chosen value both be stated explicitly.
"""

import os, sys, time
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
from record_baseline_rollout import pick_reference_agent

from discriminator_model import Discriminator
from feature_derivation import compute_raw_features
from feature_normalization import FeatureNormaliser

ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
DISC_CKPT = os.path.join(PROJECT_ROOT, "src/discriminator/discriminator_phase_b_checkpoint.pt")
NORM_PATH = os.path.join(PROJECT_ROOT, "src/discriminator/feature_normaliser.pkl")

N_EPISODES = 3          # more than one -- episode-to-episode variance in task
                         # return is large (0 vs 3 goals), one sample is not enough
MAX_STEPS = 3000
GAMMA = 0.99


def build_disc_input(o_shared, sticky_actions, action_captured, normaliser):
    step = {
        'left_team': o_shared['left_team'], 'left_team_direction': o_shared['left_team_direction'],
        'right_team': o_shared['right_team'], 'right_team_direction': o_shared['right_team_direction'],
        'ball': o_shared['ball'], 'ball_direction': o_shared['ball_direction'],
        'action_captured': action_captured,
        'sticky_actions': sticky_actions,
        'steps_left': o_shared['steps_left'],
        'score_left': o_shared['score'][0], 'score_right': o_shared['score'][1],
    }
    return normaliser.transform(compute_raw_features(step))


def run():
    actor = torch.load(ACTOR_PATH, map_location="cpu"); actor.eval()
    encoder = FeatureEncoder()

    discriminator = Discriminator()
    ckpt = torch.load(DISC_CKPT, map_location="cpu")
    discriminator.load_state_dict(ckpt["model_state_dict"])
    discriminator.eval()
    print(f"Discriminator: GATE-PASSING checkpoint, held-out acc {ckpt.get('final_held_out_acc', 'n/a'):.4f}")

    normaliser = FeatureNormaliser()
    normaliser.load(NORM_PATH)

    env = football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4, number_of_right_players_agent_controls=0,
        render=False,
    )

    per_episode = []

    for ep in range(N_EPISODES):
        raw_obs = env.reset()
        disc_feats, task_rewards = [], []
        step, done = 0, False

        while not done and step < MAX_STEPS:
            actions = []
            for i in range(4):
                state = MinimalState(n_player=5); state.set_obs(raw_obs[i])
                feat = torch.as_tensor(encoder.encode_each(state), dtype=torch.float32).unsqueeze(0)
                avail = torch.as_tensor(encoder.get_available_actions(raw_obs[i], 0.0, []), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act, _, _, _ = actor(feat, None, None, avail, True, None)   # explore=True, matches rollout collection
                actions.append(int(act.item()))

            o_shared = raw_obs[0]
            ref_agent_i, _ = pick_reference_agent(raw_obs, range(4))
            disc_feats.append(build_disc_input(
                o_shared, raw_obs[ref_agent_i]['sticky_actions'], actions[ref_agent_i], normaliser
            ))

            raw_obs, reward, done, info = env.step(actions)
            task_rewards.append(float(np.sum(reward)))
            step += 1

        with torch.no_grad():
            r_style = discriminator(torch.as_tensor(np.stack(disc_feats), dtype=torch.float32)).numpy()

        task = np.array(task_rewards)
        # UNDISCOUNTED sums -- what actually enters the objective per episode.
        # Discounted shown too since GAE uses gamma; both reported rather than
        # picking one and hoping it's the right basis.
        disc_factors = GAMMA ** np.arange(len(task))
        rec = {
            "steps": step,
            "style_sum_abs": float(np.abs(r_style).sum()),
            "style_mean": float(r_style.mean()),
            "style_sum_signed": float(r_style.sum()),
            "task_sum_abs": float(np.abs(task).sum()),
            "task_sum_signed": float(task.sum()),
            "task_sum_discounted": float((task * disc_factors).sum()),
            "style_sum_discounted": float((r_style * disc_factors).sum()),
        }
        per_episode.append(rec)
        print(f"  ep{ep}: {step} steps | style Σ|r| = {rec['style_sum_abs']:9.1f} "
              f"(mean {rec['style_mean']:+.4f}) | task Σ|r| = {rec['task_sum_abs']:.1f} "
              f"(signed {rec['task_sum_signed']:+.1f})")

    env.close()

    style_abs = np.mean([r["style_sum_abs"] for r in per_episode])
    task_abs = np.mean([r["task_sum_abs"] for r in per_episode])
    style_disc = np.mean([abs(r["style_sum_discounted"]) for r in per_episode])
    task_disc = np.mean([abs(r["task_sum_discounted"]) for r in per_episode])

    print("\n" + "=" * 70)
    print("MEASURED MAGNITUDES (mean over episodes)")
    print("=" * 70)
    print(f"  accumulated |r_style| per episode : {style_abs:10.2f}")
    print(f"  accumulated |r_task|   per episode : {task_abs:10.2f}")
    print(f"  ratio (style / task)               : {style_abs / max(task_abs, 1e-9):10.1f}x")
    print()
    print(f"  discounted |r_style| (gamma={GAMMA})  : {style_disc:10.2f}")
    print(f"  discounted |r_task|                : {task_disc:10.2f}")
    print(f"  discounted ratio                   : {style_disc / max(task_disc, 1e-9):10.1f}x")

    alpha_undiscounted = task_abs / max(style_abs, 1e-9)
    alpha_discounted = task_disc / max(style_disc, 1e-9)

    print("\n" + "=" * 70)
    print("DERIVED ALPHA")
    print("=" * 70)
    print(f"  alpha (undiscounted basis) : {alpha_undiscounted:.6f}")
    print(f"  alpha (discounted basis)   : {alpha_discounted:.6f}")
    print(f"\n  Current placeholder in finetune_objective.py: 0.1")
    print(f"  -> placeholder is off by ~{0.1 / max(alpha_undiscounted, 1e-9):.0f}x")
    print("\n  Both bases reported deliberately. The undiscounted ratio is the")
    print("  honest 'same order of magnitude' criterion the methodology asks for;")
    print("  the discounted one reflects what GAE actually propagates. If they")
    print("  differ materially, that difference is itself worth noting rather")
    print("  than silently picking one.")


if __name__ == '__main__':
    run()
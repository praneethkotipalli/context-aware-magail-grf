"""
measure_alpha_offline.py

Calibrates ALPHA on per-step VARIANCE, not accumulated sums. Run from
src/discriminator/. Needs NO GRF -- only the cached rollout .npz files,
the gate-passing discriminator, and the normaliser.

WHY VARIANCE, NOT SUMS: compute_gae ends with
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
Advantage normalisation removes any constant offset. r_style is dense and
nearly constant (~-1.0 every step); task reward is sparse and spiky (0 for
~2997 steps, +/-4 at goals). Matching their SUMS makes the style signal's
variance contribution vanish -- and variance is the only thing that
survives normalisation and reaches the policy gradient.

The earlier sum-based calibration (alpha=0.000864) is why SAP did not move.
"""

import os, glob
import numpy as np
import torch
from pathlib import Path

from feature_derivation import compute_raw_features
from feature_normalization import FeatureNormaliser
from discriminator_model import Discriminator

ROLLOUTS = os.path.join(str(Path.home()),
    "dissertation/context-aware-magail-grf/data/baseline_rollouts_for_verification")
DISC_CKPT = "discriminator_phase_b_checkpoint.pt"
NORM_PATH = "feature_normaliser.pkl"
N_EPISODES = 20        # enough for a stable std estimate without being slow


def main():
    disc = Discriminator()
    ck = torch.load(DISC_CKPT, map_location="cpu")
    disc.load_state_dict(ck["model_state_dict"]); disc.eval()
    print(f"discriminator: held-out acc {ck.get('final_held_out_acc', 'n/a')}")

    norm = FeatureNormaliser(); norm.load(NORM_PATH)

    paths = sorted(glob.glob(os.path.join(ROLLOUTS, "*.npz")))[:N_EPISODES]
    print(f"scoring {len(paths)} frozen-baseline episodes...\n")

    task_stds, style_stds, style_means = [], [], []
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        n = len(d['steps_left'])
        feats = np.stack([
            norm.transform(compute_raw_features({
                'left_team': d['left_team'][t],
                'left_team_direction': d['left_team_direction'][t],
                'right_team': d['right_team'][t],
                'right_team_direction': d['right_team_direction'][t],
                'ball': d['ball'][t], 'ball_direction': d['ball_direction'][t],
                'action_captured': d['action_captured'][t],
                'sticky_actions': d['sticky_actions'][t],
                'steps_left': d['steps_left'][t],
                'score_left': d['score_left'][t], 'score_right': d['score_right'][t],
            })) for t in range(n)
        ])
        with torch.no_grad():
            rs = disc(torch.as_tensor(feats, dtype=torch.float32)).numpy()

        # shared team reward x4 agents -- matches env.step()'s np.sum(reward).
        # Verified against the pilot: a 2-1 game summed to 4.0.
        task = 4.0 * (np.diff(d['score_left'].astype(float))
                      - np.diff(d['score_right'].astype(float)))

        task_stds.append(task.std())
        style_stds.append(rs.std())
        style_means.append(rs.mean())

    t_std = float(np.mean(task_stds))
    s_std = float(np.mean(style_stds))
    s_mean = float(np.mean(style_means))
    alpha = t_std / max(s_std, 1e-9)

    print("=" * 62)
    print(f"  per-step task  std : {t_std:.5f}")
    print(f"  per-step style std : {s_std:.5f}   (mean {s_mean:+.4f})")
    print("=" * 62)
    print(f"  ALPHA (variance-matched) = {alpha:.4f}")
    print()
    print(f"  vs current placeholder 0.25 : off by {alpha/0.25:.2f}x")
    print(f"  vs original 0.001           : off by {alpha/0.001:.0f}x")
    print()
    print("  Set ALPHA in finetune_objective.py to the value above.")
    print("  This is a MEASURED number -- report it and the underlying")
    print("  std ratio explicitly in the methodology, per the requirement")
    print("  that alpha's calibration be stated rather than assumed.")


if __name__ == '__main__':
    main()
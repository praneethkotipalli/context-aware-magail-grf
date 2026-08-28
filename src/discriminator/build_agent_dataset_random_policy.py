"""
build_agent_dataset_random_policy.py

Converts raw random-policy rollout .npz files into the same
(features, bins) cache format as expert_features_cache.npz, so
BalancedContextSampler can wrap agent data identically to expert data.

Reuses classify_bin() from build_expert_dataset.py directly -- same
taxonomy, not a second parallel definition that could drift.

No episode_outcomes / episode split needed here: the 75%-held-out gate
applies only to the EXPERT side (Section 3.3.3's "held-out demonstration
split"). The agent side has no train/held-out distinction at any point.
"""

import os, glob
import numpy as np
from pathlib import Path

#from feature_derivation import compute_raw_features
#from feature_normalization import normalize_features
#from build_expert_dataset import classify_bin

from feature_derivation import compute_raw_features
from feature_normalization import FeatureNormaliser
from build_expert_dataset import classify_bin

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
EPISODES_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "random_policy_rollouts")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "random_policy_features_cache.npz")


"""def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))
    print(f"Building random-policy feature cache from {len(paths)} episodes...")

    all_features, all_bins = [], []

    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        n = len(d['steps_left'])
        for t in range(n):
            step = {
                'left_team': d['left_team'][t], 'left_team_direction': d['left_team_direction'][t],
                'right_team': d['right_team'][t], 'right_team_direction': d['right_team_direction'][t],
                'ball': d['ball'][t], 'ball_direction': d['ball_direction'][t],
                'action_captured': d['action_captured'][t],
                'steps_left': d['steps_left'][t], 'score_left': d['score_left'][t], 'score_right': d['score_right'][t],
            }
            raw = compute_raw_features(step)
            norm = normalize_features(raw)
            t_norm = d['steps_left'][t] / 3001.0
            delta_score = int(d['score_left'][t]) - int(d['score_right'][t])
            all_features.append(norm)
            all_bins.append(classify_bin(t_norm, delta_score))

    features = np.stack(all_features).astype(np.float32)
    bins = np.array(all_bins, dtype=np.int64)

    np.savez_compressed(CACHE_PATH, features=features, bins=bins)
    print(f"Cached {features.shape[0]} steps, {features.shape[1]} dims -> {CACHE_PATH}")

    print("\nCell distribution (0-8 = early/mid/late x win/loss/draw):")
    for b in range(9):
        tz, sz = b // 3, b % 3
        tz_name = ['early', 'mid', 'late'][tz]
        sz_name = ['win', 'loss', 'draw'][sz]
        count = (bins == b).sum()
        print(f"  cell {b} ({tz_name}/{sz_name}): {count} steps")


if __name__ == '__main__':
    run()"""
def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))
    print(f"Building random feature cache from {len(paths)} episodes...")

    all_raw_features, all_bins = [], []

    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        n = len(d['steps_left'])
        for t in range(n):
            step = {
                'left_team': d['left_team'][t], 'left_team_direction': d['left_team_direction'][t],
                'right_team': d['right_team'][t], 'right_team_direction': d['right_team_direction'][t],
                'ball': d['ball'][t], 'ball_direction': d['ball_direction'][t],
                'action_captured': d['action_captured'][t],
                'sticky_actions': d['sticky_actions'][t],  # <-- CRITICAL: Added for the new 139-dim features
                'steps_left': d['steps_left'][t], 'score_left': d['score_left'][t], 'score_right': d['score_right'][t],
            }
            # Calculate raw features only during the loop
            raw = compute_raw_features(step)
            
            t_norm = d['steps_left'][t] / 3001.0
            delta_score = int(d['score_left'][t]) - int(d['score_right'][t])
            
            all_raw_features.append(raw)
            all_bins.append(classify_bin(t_norm, delta_score))

    # Convert to a massive 2D array
    raw_features_array = np.stack(all_raw_features).astype(np.float32)

    # LOAD the frozen normalizer to prevent distribution leakage
    norm_path = os.path.join(os.path.dirname(__file__), "feature_normaliser.pkl")
    normaliser = FeatureNormaliser()
    normaliser.load(norm_path)
    
    # Vectorized transformation using the frozen stats
    features = normaliser.transform(raw_features_array)
    bins = np.array(all_bins, dtype=np.int64)

    np.savez_compressed(CACHE_PATH, features=features, bins=bins)
    print(f"Cached {features.shape[0]} steps, {features.shape[1]} dims -> {CACHE_PATH}")

    print("\nCell distribution (0-8 = early/mid/late x win/loss/draw):")
    for b in range(9):
        tz, sz = b // 3, b % 3
        tz_name = ['early', 'mid', 'late'][tz]
        sz_name = ['win', 'loss', 'draw'][sz]
        count = (bins == b).sum()
        print(f"  cell {b} ({tz_name}/{sz_name}): {count} steps")


if __name__ == '__main__':
    run()
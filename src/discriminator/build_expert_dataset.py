# src/discriminator/build_expert_dataset.py
import os, glob
import numpy as np
from pathlib import Path
from feature_derivation import compute_raw_features
from feature_normalization import normalize_features

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
EPISODES_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations", "episodes")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "expert_features_cache.npz")


def classify_bin(t_norm, delta_score):
    """Same 9-cell taxonomy as the locked context_region analysis --
    one canonical scheme reused here, not a second parallel one."""
    elapsed_min = (1 - t_norm) * 90.0
    if elapsed_min < 20: time_zone = 0     # early
    elif elapsed_min < 70: time_zone = 1   # mid
    else: time_zone = 2                     # late

    if delta_score > 0: score_zone = 0      # win
    elif delta_score < 0: score_zone = 1    # loss
    else: score_zone = 2                     # draw

    return time_zone * 3 + score_zone        # 0..8, 9 cells total


def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))
    print(f"Building expert feature cache from {len(paths)} episodes...")

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
    run()
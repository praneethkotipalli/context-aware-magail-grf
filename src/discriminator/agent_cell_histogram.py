"""agent_cell_histogram.py -- now takes a directory, checks any rollout source."""
import os, sys, glob
import numpy as np
from build_expert_dataset import classify_bin

def run(rollout_dir):
    paths = sorted(glob.glob(os.path.join(rollout_dir, "*.npz")))
    print(f"Checking cell coverage across {len(paths)} episodes in {rollout_dir}...")

    bins = []
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        n = len(d['steps_left'])
        for t in range(n):
            t_norm = d['steps_left'][t] / 3001.0
            delta_score = int(d['score_left'][t]) - int(d['score_right'][t])
            bins.append(classify_bin(t_norm, delta_score))
    bins = np.array(bins)

    print(f"\n{len(bins)} total steps\n")
    empty_cells = []
    for b in range(9):
        tz, sz = b // 3, b % 3
        tz_name = ['early', 'mid', 'late'][tz]
        sz_name = ['win', 'loss', 'draw'][sz]
        count = int((bins == b).sum())
        flag = "  <-- EMPTY" if count == 0 else ("  <-- THIN" if count < 20 else "")
        print(f"  cell {b} ({tz_name}/{sz_name}): {count} steps{flag}")
        if count == 0:
            empty_cells.append(f"{tz_name}/{sz_name}")

    print()
    if empty_cells:
        print(f"WARNING: {len(empty_cells)} cell(s) empty: {empty_cells}")
    else:
        print("All 9 cells populated.")
    return bins

if __name__ == '__main__':
    rollout_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "dissertation", "context-aware-magail-grf",
        "data", "baseline_rollouts_for_verification"
    )
    run(rollout_dir)
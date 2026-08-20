# find_ball_z_outlier.py
import glob, os
import numpy as np
from pathlib import Path

EPISODES_DIR = os.path.expanduser("~/dissertation/context-aware-magail-grf/data/demonstrations/episodes")

for p in sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz"))):
    d = dict(np.load(p, allow_pickle=True))
    z = d['ball'][:, 2]
    if z.max() > 1.0:  # anything above ~1.0 is already well outside normal ball-height range
        idx = int(np.argmax(z))
        print(f"{os.path.basename(p)}: max ball_z={z.max():.4f} at step {idx}")
        print(f"  ball at that step: {d['ball'][idx]}")
        print(f"  ball_direction at that step: {d['ball_direction'][idx]}")
        print(f"  steps_left: {d['steps_left'][idx]}  score: {d['score_left'][idx]}-{d['score_right'][idx]}")
        # show a small window around it -- is it one freak frame or a sustained sequence?
        lo, hi = max(0, idx-3), min(len(z), idx+4)
        print(f"  z values around it (steps {lo}-{hi-1}): {z[lo:hi]}")
        print()
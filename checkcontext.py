import numpy as np
import glob

# find the episode(s) that contain late_winning steps
for path in sorted(glob.glob("data/demonstrations/episodes/*.npz")):
    d = np.load(path, allow_pickle=True)
    regions = d['context_region']
    mask = regions == 'late_winning'
    if mask.sum() > 0:
        print(f"\n{path}")
        print(f"  late_winning steps: {mask.sum()}")
        sl = d['score_left'][mask]
        sr = d['score_right'][mask]
        print(f"  score_left range at those steps:  {sl.min()}-{sl.max()}")
        print(f"  score_right range at those steps: {sr.min()}-{sr.max()}")
        print(f"  score_left > score_right for all such steps: {(sl > sr).all()}")
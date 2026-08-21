"""inspect_expert_cache.py -- check the real cache's structure before wiring it in."""
import numpy as np

cache = np.load("expert_features_cache.npz")
print("keys:", list(cache.keys()))
for k in cache.keys():
    arr = cache[k]
    print(f"  {k}: shape={arr.shape}  dtype={arr.dtype}")
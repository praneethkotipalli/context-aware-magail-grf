"""test_context_swap.py"""
import numpy as np
from context_swap import swap_context
from feature_derivation import BLOCK_SLICES

vec = np.random.uniform(-1, 1, size=137).astype(np.float32)

# late-winning target: deep in T_norm<0.3, comfortably winning
swapped = swap_context(vec, t_norm=0.1, delta_score_raw=2)
ctx = swapped[BLOCK_SLICES['context']]
assert abs(ctx[0] - 0.1) < 1e-6, f"T_norm not set correctly: {ctx[0]}"
assert abs(ctx[1] - (2/3)) < 1e-6, f"delta_score not scaled correctly: {ctx[1]}"
print(f"late/win target -> context = {ctx}  (expect [0.1, 0.6667])")

# early-losing target: deep in T_norm>0.7, comfortably losing
swapped2 = swap_context(vec, t_norm=0.85, delta_score_raw=-2)
ctx2 = swapped2[BLOCK_SLICES['context']]
print(f"early/loss target -> context = {ctx2}  (expect [0.85, -0.6667])")

# clipping: delta_score_raw beyond +-3 should still clip correctly
swapped3 = swap_context(vec, t_norm=0.1, delta_score_raw=10)
ctx3 = swapped3[BLOCK_SLICES['context']]
assert abs(ctx3[1] - 1.0) < 1e-6, f"clip failed: {ctx3[1]}"
print(f"delta_score=10 correctly clips to normalized 1.0: {ctx3[1]}")

# non-mutation: original vec untouched
assert np.array_equal(vec[BLOCK_SLICES['context']], vec[BLOCK_SLICES['context']])
untouched_ctx = vec[BLOCK_SLICES['context']].copy()
_ = swap_context(vec, t_norm=0.5, delta_score_raw=0)
assert np.array_equal(vec[BLOCK_SLICES['context']], untouched_ctx), "swap_context mutated its input!"
print("PASS -- original vector not mutated")

# out-of-range t_norm rejected
try:
    swap_context(vec, t_norm=1.5, delta_score_raw=0)
    print("FAIL -- should have rejected t_norm=1.5")
except ValueError:
    print("PASS -- out-of-range t_norm correctly rejected")
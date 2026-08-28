"""test_context_shift_scoring.py"""
"""import numpy as np
import torch.nn as nn

from context_shift_scoring import select_by_true_context, score_context_shift, LATE_WINNING, EARLY_LOSING
from feature_derivation import BLOCK_SLICES

INPUT_DIM = 137
ctx_slice = BLOCK_SLICES['context']

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

np.random.seed(0)
N = 1000
features = np.random.uniform(-1, 1, size=(N, INPUT_DIM)).astype(np.float32)
t_norm = np.random.uniform(0, 1, size=N)
delta_score_norm = np.random.uniform(-1, 1, size=N)
features[:, ctx_slice] = np.stack([t_norm, delta_score_norm], axis=1)

mask = select_by_true_context(features, t_norm_max=LATE_WINNING['t_norm_max'],
                               delta_score_sign=LATE_WINNING['delta_score_sign'])
expected = (t_norm <= 0.3) & (delta_score_norm > 0)
check("late-winning mask matches hand-computed selection exactly", np.array_equal(mask, expected))

mask2 = select_by_true_context(features, t_norm_min=EARLY_LOSING['t_norm_min'],
                                delta_score_sign=EARLY_LOSING['delta_score_sign'])
expected2 = (t_norm >= 0.7) & (delta_score_norm < 0)
check("early-losing mask matches hand-computed selection exactly", np.array_equal(mask2, expected2))

class ContextOnlyDiscriminator(nn.Module):
    ""Logit depends ONLY on delta_score -- makes the expected shift
    direction hand-verifiable, not trusted from an opaque model.""
    def forward(self, x):
        return (x[:, ctx_slice][:, 1] * 10.0).unsqueeze(1)

disc = ContextOnlyDiscriminator()
subset = features[mask][:20]
true_probs, swapped_probs, shifts = score_context_shift(
    disc, subset, target_t_norm=0.85, target_delta_score_raw=-2
)

check("true/swapped/shifts all length 20", len(true_probs) == len(swapped_probs) == len(shifts) == 20)
check("all probabilities in (0,1)",
      np.all((true_probs > 0) & (true_probs < 1)) and np.all((swapped_probs > 0) & (swapped_probs < 1)))
check("swapping to a losing context decreases P(expert) for every example",
      np.all(shifts < 0))
print(f"  mean shift = {shifts.mean():.4f}")

print("\nAll checks passed.")"""

import numpy as np
import torch.nn as nn

from context_shift_scoring import select_by_true_context, score_context_shift, LATE_WINNING, LATE_LOSING
from feature_derivation import BLOCK_SLICES

INPUT_DIM = 139
ctx_slice = BLOCK_SLICES['context']

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

np.random.seed(0)
N = 1000
features = np.random.uniform(-1, 1, size=(N, INPUT_DIM)).astype(np.float32)
t_norm = np.random.uniform(0, 1, size=N)
delta_score_norm = np.random.uniform(-1, 1, size=N)
features[:, ctx_slice] = np.stack([t_norm, delta_score_norm], axis=1)

mask = select_by_true_context(features, t_norm_max=LATE_WINNING['t_norm_max'],
                               delta_score_sign=LATE_WINNING['delta_score_sign'])
expected = (t_norm <= 0.3) & (delta_score_norm > 0)
check("late-winning mask matches hand-computed selection exactly", np.array_equal(mask, expected))

mask2 = select_by_true_context(features, t_norm_max=LATE_LOSING['t_norm_max'],
                                delta_score_sign=LATE_LOSING['delta_score_sign'])
expected2 = (t_norm <= 0.3) & (delta_score_norm < 0)
check("late-losing mask matches hand-computed selection exactly", np.array_equal(mask2, expected2))

class ContextOnlyDiscriminator(nn.Module):
    """Logit depends ONLY on delta_score -- makes the expected shift
    direction hand-verifiable, not trusted from an opaque model."""
    def forward(self, x):
        return (x[:, ctx_slice][:, 1] * 10.0).unsqueeze(1)

disc = ContextOnlyDiscriminator()
subset = features[mask][:20]
true_probs, swapped_probs, shifts = score_context_shift(
    disc, subset, target_t_norm=0.1, target_delta_score_raw=-2
)

check("true/swapped/shifts all length 20", len(true_probs) == len(swapped_probs) == len(shifts) == 20)
check("all probabilities in (0,1)",
      np.all((true_probs > 0) & (true_probs < 1)) and np.all((swapped_probs > 0) & (swapped_probs < 1)))
check("swapping to a losing context decreases P(expert) for every example",
      np.all(shifts < 0))
print(f"  mean shift = {shifts.mean():.4f}")

print("\nAll checks passed.")
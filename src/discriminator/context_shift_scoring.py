"""
context_shift_scoring.py

Scores the counterfactual context shift for the swap-test gate (Section
3.3.3): take real expert examples whose TRUE context matches one regime,
swap to the OPPOSITE regime, measure how much P(expert) moves.

Does NOT decide pass/fail or aggregation -- that's next.
"""

"""import numpy as np
import torch

from context_swap import swap_context
from feature_derivation import BLOCK_SLICES

# Locked definitions, POST-bug-fix direction (T_norm starts near 1.0,
# DECREASES as the match proceeds):
LATE_WINNING = dict(t_norm_max=0.3, delta_score_sign=+1)
EARLY_LOSING = dict(t_norm_min=0.7, delta_score_sign=-1)


def select_by_true_context(features, t_norm_min=None, t_norm_max=None, delta_score_sign=None):
    ""
    features: (N, 137) ALREADY-NORMALIZED vectors (e.g. straight from
        expert_features_cache.npz). Filters using the context slot
        embedded in each vector directly -- no separate raw storage
        needed, since normalize_features() leaves T_norm untouched and
        only rescales delta_score by a positive constant (sign preserved).

    Returns a boolean mask, not a filtered array.
    ""
    ctx_slice = BLOCK_SLICES['context']
    t_norm = features[:, ctx_slice][:, 0]
    delta_score_norm = features[:, ctx_slice][:, 1]

    mask = np.ones(features.shape[0], dtype=bool)
    if t_norm_min is not None:
        mask &= (t_norm >= t_norm_min)
    if t_norm_max is not None:
        mask &= (t_norm <= t_norm_max)
    if delta_score_sign is not None:
        if delta_score_sign > 0:
            mask &= (delta_score_norm > 0)
        elif delta_score_sign < 0:
            mask &= (delta_score_norm < 0)
        else:
            mask &= (delta_score_norm == 0)
    return mask


def score_context_shift(discriminator, features, target_t_norm, target_delta_score_raw):
    ""
    features: (N, 137) already-normalized, ALREADY FILTERED to the
        population being swapped away from.
    target_t_norm, target_delta_score_raw: RAW targets (same convention
        as swap_context -- delta_score un-clipped, e.g. -2 not -0.6667).

    Returns (true_probs, swapped_probs, shifts), all (N,).
    shifts = swapped_probs - true_probs, SIGNED (not abs()) -- direction
    is itself diagnostic: an unexpectedly-signed result flags something
    worth investigating, not noise to discard.
    ""
    swapped = np.stack([
        swap_context(row, target_t_norm, target_delta_score_raw) for row in features
    ])

    discriminator.eval()
    with torch.no_grad():
        true_t = torch.as_tensor(features, dtype=torch.float32)
        swapped_t = torch.as_tensor(swapped, dtype=torch.float32)
        true_probs = torch.sigmoid(discriminator(true_t).view(-1)).numpy()
        swapped_probs = torch.sigmoid(discriminator(swapped_t).view(-1)).numpy()

    return true_probs, swapped_probs, swapped_probs - true_probs"""
import numpy as np
import torch

from context_swap import swap_context
from feature_derivation import BLOCK_SLICES

# Locked definitions, POST-bug-fix direction (T_norm starts near 1.0,
# DECREASES as the match proceeds):
LATE_WINNING = dict(t_norm_max=0.2222, delta_score_sign=+1)
LATE_LOSING = dict(t_norm_max=0.2222, delta_score_sign=-1)


def select_by_true_context(features, t_norm_min=None, t_norm_max=None, delta_score_sign=None):
    """
    features: (N, 139) ALREADY-NORMALIZED vectors (e.g. straight from
        expert_features_cache.npz). Filters using the context slot
        embedded in each vector directly -- no separate raw storage
        needed, since normalize_features() leaves T_norm untouched and
        only rescales delta_score by a positive constant (sign preserved).

    Returns a boolean mask, not a filtered array.
    """
    ctx_slice = BLOCK_SLICES['context']
    t_norm = features[:, ctx_slice][:, 0]
    delta_score_norm = features[:, ctx_slice][:, 1]

    mask = np.ones(features.shape[0], dtype=bool)
    if t_norm_min is not None:
        mask &= (t_norm >= t_norm_min)
    if t_norm_max is not None:
        mask &= (t_norm <= t_norm_max)
    if delta_score_sign is not None:
        if delta_score_sign > 0:
            mask &= (delta_score_norm > 0)
        elif delta_score_sign < 0:
            mask &= (delta_score_norm < 0)
        else:
            mask &= (delta_score_norm == 0)
    return mask


def score_context_shift(discriminator, features, target_t_norm, target_delta_score_raw):
    """
    features: (N, 139) already-normalized, ALREADY FILTERED to the
        population being swapped away from.
    target_t_norm, target_delta_score_raw: RAW targets (same convention
        as swap_context -- delta_score un-clipped, e.g. -2 not -0.6667).

    Returns (true_probs, swapped_probs, shifts), all (N,).
    shifts = swapped_probs - true_probs, SIGNED (not abs()) -- direction
    is itself diagnostic: an unexpectedly-signed result flags something
    worth investigating, not noise to discard.
    """
    swapped = np.stack([
        swap_context(row, target_t_norm, target_delta_score_raw) for row in features
    ])

    discriminator.eval()
    with torch.no_grad():
        true_t = torch.as_tensor(features, dtype=torch.float32)
        swapped_t = torch.as_tensor(swapped, dtype=torch.float32)
        true_probs = torch.sigmoid(discriminator(true_t).view(-1)).numpy()
        swapped_probs = torch.sigmoid(discriminator(swapped_t).view(-1)).numpy()

    return true_probs, swapped_probs, swapped_probs - true_probs

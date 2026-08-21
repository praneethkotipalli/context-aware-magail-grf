"""
context_swap.py

Constructs a counterfactual context for the discriminator's swap-test gate
(Section 3.3.3): take a real, already-normalized 137-dim feature vector and
replace ONLY its context slot (Block 10 -- T_norm, ΔScore) with a different
context, leaving all 135 spatial/action dims untouched.

Callers always supply RAW values here (T_norm in [0,1], ΔScore un-clipped),
matching feature_normalization.py's own convention -- this function applies
the identical clip(-3,3)/3 scaling internally, so there is exactly one place
in the codebase that knows how ΔScore gets normalized.
"""

import numpy as np
from feature_derivation import BLOCK_SLICES


def swap_context(feature_vec, t_norm, delta_score_raw):
    """
    feature_vec: a single already-normalized (137,) feature vector.
    t_norm: raw T_norm target in [0,1] -- unscaled by normalize_features,
        so this is used as-is. Recall the locked convention: T_norm starts
        near 1.0 and DECREASES as the match proceeds, so "late" match = low
        T_norm, "early" match = high T_norm.
    delta_score_raw: raw ΔScore target (e.g. +2, -2) -- NOT pre-divided by
        3. This function applies the same clip(-3, 3) / 3.0 normalize_features
        uses, so callers never touch the normalized number directly.

    Returns a NEW array -- does not mutate feature_vec.
    """
    if not (0.0 <= t_norm <= 1.0):
        raise ValueError(f"t_norm must be in [0, 1], got {t_norm}")

    swapped = feature_vec.copy()
    ctx_slice = BLOCK_SLICES['context']
    swapped[ctx_slice] = [
        t_norm,
        np.clip(delta_score_raw, -3, 3) / 3.0,
    ]
    return swapped
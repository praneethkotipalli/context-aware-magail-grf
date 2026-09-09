"""
stratified_sampler18.py

Extends context_balanced_sampler.py's 9-cell scheme to 18 cells by
crossing each context cell with sprint state, so D_int's BCE batches
have matched sprint marginals on both sides -- removes dim 135 as a
usable classification shortcut for D_int specifically (D_marg keeps
the original unbalanced 9-cell sampler unchanged, since it's SUPPOSED
to use the sprint signal).

Measured cell populations this is built against (expert side, 156,052
steps, computed via classify_bin's time_zone*3+score_zone scheme,
confirmed: cell 6 = late/win, cell 7 = late/loss):

  joint      cell          sprint  steps    reuse @ 20k iters (sqrt target)
  0  early/win   off      2238    ~38x
  1  early/win   on        119    ~163x   <- THIN, floor triggers
  2  early/loss  off      4153
  3  early/loss  on        367    ~93x
  ...
  12 late/win    off      7086
  13 late/win    on        801    ~63x
  14 late/loss   off      8103
  15 late/loss   on       6722

Only joint cell 1 (early/win x sprint=on) is thin enough to require a
floor. MIN_CELL_STEPS below merges it into its parent 9-cell's sprint=off
population for STRATIFICATION PURPOSES ONLY when a run's actual episode
cache falls under the threshold -- state this explicitly if it triggers,
since 119 steps at 163x reuse is memorization, not learning.
"""

import numpy as np

N_JOINT_CELLS = 18
MIN_CELL_STEPS = 300          # below this, sqrt-target reuse exceeds ~50x -- merge
SPRINT_DIM = 135

BASE_NAMES = [
    'early/win', 'early/loss', 'early/draw',
    'mid/win', 'mid/loss', 'mid/draw',
    'late/win', 'late/loss', 'late/draw',
]
JOINT_NAMES = [f"{BASE_NAMES[c]} spr={'on' if s else 'off'}"
              for c in range(9) for s in (0, 1)]


def joint_cell(base_cell, sprint_state):
    """base_cell: 0..8 from build_expert_dataset.classify_bin.
    sprint_state: scalar or array, thresholded at 0.5 (matches how
    ContextConditionedActor and the corpus stats treat sticky[8])."""
    return base_cell * 2 + (np.asarray(sprint_state) > 0.5).astype(int)


def build_joint_bins(base_bins, sprint_values, min_cell_steps=MIN_CELL_STEPS):
    """base_bins: (N,) int in 0..8. sprint_values: (N,) float, the raw
    (already-normalized) dim-135 value for each row.

    Returns (joint_bins, merge_report). Any joint cell with fewer than
    min_cell_steps real examples is merged into its sprint=off sibling
    (cell 2c instead of 2c+1) -- sprint=off is chosen as the merge target
    because every thin cell observed in this corpus is a sprint=on cell
    (humans rarely sprint when comfortable), so merging toward off keeps
    the larger, better-populated side as the stratification anchor.
    """
    joint = joint_cell(base_bins, sprint_values)
    counts = np.array([(joint == c).sum() for c in range(N_JOINT_CELLS)])
    merge_report = []
    for c in range(0, N_JOINT_CELLS, 2):       # c = sprint-off cell of each pair
        on_cell = c + 1
        if 0 < counts[on_cell] < min_cell_steps:
            joint = np.where(joint == on_cell, c, joint)
            merge_report.append({
                'merged_cell': JOINT_NAMES[on_cell], 'into': JOINT_NAMES[c],
                'n_examples': int(counts[on_cell]),
            })
    return joint, merge_report


def sqrt_scaled_target_18(cell_counts):
    """Same formula as the 9-cell version (sqrt of count, normalized),
    applied post-merge so no surviving cell is pathologically thin."""
    counts = np.asarray(cell_counts, dtype=float)
    if np.any(counts <= 0):
        raise ValueError(f"all surviving joint cells must be non-empty; got {counts}")
    w = np.sqrt(counts)
    return w / w.sum()


def print_joint_report(joint_bins, name='unnamed'):
    print(f"\n[{name}] 18-cell joint distribution (post-merge):")
    present = sorted(set(joint_bins.tolist()))
    for c in present:
        n = int((joint_bins == c).sum())
        label = JOINT_NAMES[c] if c < N_JOINT_CELLS else f"merged->{c}"
        print(f"  cell {c:2d} ({label:20s}): {n:7d} steps")
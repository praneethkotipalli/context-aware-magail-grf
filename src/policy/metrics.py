"""
metrics.py

SAP, win rate, MECHA, and a CSI PROXY computed from natural rollout data.

IMPORTANT GAP, stated explicitly rather than papered over: Section 3.5.1's
locked CSI definition specifies "controlled scenarios that fix (T_norm,
delta_score) at the four match regions" -- i.e. deliberately scripted
match starts at specific game states, not naturally-occurring rollout
steps. No such controlled-scenario GRF harness has been built anywhere in
this project so far. What's implemented below is a PROXY: CSI computed by
binning NATURALLY-OCCURRING steps by their true (T_norm, delta_score),
same approach the counterfactual gate and oracle scripts already use.
This is useful for monitoring convergence during training (is behaviour
differentiating by context AT ALL as training proceeds), but is NOT a
substitute for the methodology's controlled-scenario protocol, which is
a separate, not-yet-built piece needed before the final headline CSI
result can be reported. Flag this gap explicitly if these convergence
numbers are used in any dissertation figure.
"""

import numpy as np

PITCH_AREA = 2.0 * 0.84   # x in [-1,1], y in [-0.42,0.42] -- matches
                            # PITCH_DIAG's implied dimensions elsewhere
SPRINT_STICKY_INDEX = 8


def compute_sap(sticky_actions: np.ndarray) -> float:
    """sticky_actions: (T, >=9). Returns percentage, matching the
    project's own locked definition (verified via verify_sap_definition.py
    to match documented figures, NOT action_captured==13)."""
    return 100.0 * float(np.mean(sticky_actions[:, SPRINT_STICKY_INDEX] == 1))


def compute_win_rate(final_scores: list) -> dict:
    """final_scores: list of (score_left, score_right) tuples, one per
    evaluation episode. Returns win/draw/loss rates, NOT just a single
    number -- the draw case matters (per the corpus's own 14W/13D/25L
    finding: win-vs-everything-else, not a smooth gradient)."""
    n = len(final_scores)
    wins = sum(1 for l, r in final_scores if l > r)
    draws = sum(1 for l, r in final_scores if l == r)
    losses = n - wins - draws
    return {
        "win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n,
        "n_episodes": n,
    }


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, O(n log n). points: (N, 2). Returns hull
    vertices in counter-clockwise order. Standard algorithm, included
    because 5 player positions are not guaranteed to already be in convex
    position / hull order -- assuming they are would silently give a
    wrong (possibly negative or non-convex) area from the shoelace formula."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _shoelace_area(hull: np.ndarray) -> float:
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_mecha(left_team_positions: np.ndarray, has_possession: np.ndarray) -> float:
    """
    left_team_positions: (T, 5, 2). has_possession: (T,) bool -- ball_owned_team==0,
    matching the field already recorded in every rollout .npz.

    Restricted to possession timesteps ONLY, per the locked definition --
    defensive compression is tactically correct and penalizing it via
    formation-spread would confound the metric's meaning.
    """
    poss_idx = np.where(has_possession)[0]
    if len(poss_idx) == 0:
        return float('nan')   # no possession timesteps -- report as missing, not 0

    areas = []
    for t in poss_idx:
        hull = _convex_hull(left_team_positions[t])
        areas.append(_shoelace_area(hull) / PITCH_AREA)
    return float(np.mean(areas))


def compute_csi_proxy(t_norm_array, delta_score_array, metric_array,
                       t_norm_max=0.2222):
    """
    NATURAL-ROLLOUT PROXY -- see module docstring. metric_array is a
    PER-STEP quantity (e.g. sticky_actions[:,8] for a SAP-based proxy)
    aligned with t_norm_array/delta_score_array.

    Returns (csi_signed, n_late_winning, n_late_losing). signed per the
    locked convention: M(late-winning) - M(late-losing). None if either
    population is empty this evaluation window.
    """
    lw_mask = (t_norm_array <= t_norm_max) & (delta_score_array > 0)
    ll_mask = (t_norm_array <= t_norm_max) & (delta_score_array < 0)

    n_lw, n_ll = int(lw_mask.sum()), int(ll_mask.sum())
    if n_lw == 0 or n_ll == 0:
        return None, n_lw, n_ll

    csi = float(metric_array[lw_mask].mean() - metric_array[ll_mask].mean())
    return csi, n_lw, n_ll
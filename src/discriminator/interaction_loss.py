"""
interaction_loss.py

The core fix for B2 (main-effect / interaction confusion). Adds a
difference-in-differences hinge to the discriminator loss, targeting
the quantity CSI actually needs: a change in the ACTION ranking across
context, not just a shift in overall output level (which is what the
existing swap-consistency hinge in discriminator_loss.py trains, and
which GAE removes as a state-dependent baseline -- see B2/B3 in the
remediation plan).

CONFIRMED CONSTANTS (do not re-derive, these are measured):
  cell 6 = late/win   (classify_bin: time_zone=2, score_zone=0)
  cell 7 = late/loss  (classify_bin: time_zone=2, score_zone=1)
  P(sprint | expert, cell 7) = 0.4534
  P(sprint | expert, cell 6) = 0.1016
  P(sprint | agent,  cell 7) = 0.8892
  P(sprint | agent,  cell 6) = 0.8777
  expert log-odds diff (LL - LW) = +1.9926
  agent  log-odds diff (LL - LW) = +0.1118   <- confirmed NOT zero
  ORACLE DiD (expert - agent)    = +1.8809
  GATE THRESHOLD = 0.30 * 1.8809 = 0.5643    (pre-registered, 30% of ceiling)

Training uses SYNTHETIC paired states (dim 135 overwritten directly) for
a clean per-sample gradient. The GATE below uses only REAL held-out
states in four disjoint groups -- synthesis is fine for gradient, not
for a pass/fail criterion (avoids evaluating off-manifold states: a
synthetic "sprinting at walking velocity" row is not evidence of
anything about real discriminator behaviour).
"""

import numpy as np
import torch

SPRINT_DIM = 135
T_LATE = 0.1          # matches context_shift_scoring's LATE_* t_norm_max convention
DS_LOSE_RAW = -2
DS_WIN_RAW = +2

ORACLE_DID = 1.8809
GATE_THRESHOLD = 0.30 * ORACLE_DID     # 0.5643, pre-registered before Phase 1 runs

DEFAULT_GAMMA_INT = 6.0        # same starting point as gamma_swap; re-check with
                                 # a magnitude test before trusting on a real run
DEFAULT_INT_MARGIN = 0.80      # above the 0.5643 gate for buffer, same logic as
                                 # swap_margin=0.15 sitting above the 0.1 swap gate


def _swap_context_and_sprint(row, t_norm, delta_score_raw, sprint_val):
    """Requires context_swap.swap_context (existing, sets dims 137:139).
    Overwrites dim 135 on top of that. Import kept local to avoid a hard
    dependency if this module is used standalone for the gate only."""
    from context_swap import swap_context
    out = swap_context(row, t_norm, delta_score_raw).copy()
    out[SPRINT_DIM] = sprint_val
    return out


def interaction_hinge(discriminator, base_features, rng, batch_size=64,
                      margin=DEFAULT_INT_MARGIN):
    """Training-time term. base_features: (N,139) real expert states,
    TRAIN split only (mirrors swap_lw_features/swap_ll_features -- never
    held-out). Returns (hinge, did_mean) or (None, None) if empty.

    did = [D(spr=1,late-lose) - D(spr=0,late-lose)]
        - [D(spr=1,late-win)  - D(spr=0,late-win)]

    An additive discriminator f(s,a) + g(c) gives did == 0 exactly, by
    construction -- that is the whole point of the DiD form: it cannot
    be satisfied by a context main effect, only by a genuine interaction.
    """
    if base_features is None or len(base_features) == 0:
        return None, None
    n = min(batch_size, len(base_features))
    idx = rng.choice(len(base_features), size=n, replace=False)
    base = base_features[idx]

    def variant(sprint_val, t_norm, ds_raw):
        v = np.stack([_swap_context_and_sprint(r, t_norm, ds_raw, sprint_val) for r in base])
        return torch.as_tensor(v, dtype=torch.float32)

    d1_l = discriminator(variant(1.0, T_LATE, DS_LOSE_RAW)).view(-1)
    d0_l = discriminator(variant(0.0, T_LATE, DS_LOSE_RAW)).view(-1)
    d1_w = discriminator(variant(1.0, T_LATE, DS_WIN_RAW)).view(-1)
    d0_w = discriminator(variant(0.0, T_LATE, DS_WIN_RAW)).view(-1)

    did = (d1_l - d0_l) - (d1_w - d0_w)
    hinge = torch.relu(margin - did).mean()
    return hinge, did.mean().detach()


def interaction_gate(discriminator, held_feat, held_bins, threshold=GATE_THRESHOLD,
                     min_group_size=50):
    """PRIMARY gate. Real held-out states only, four disjoint groups, no
    synthesis. Returns (passed, did_value, group_sizes, warning).

    If any group falls below min_group_size, the result is still
    returned but flagged -- report on train+held-out combined with that
    stated, per the remediation plan, rather than trusting a noisy
    held-out-only number silently.
    """
    def group(cell, sprint_on):
        m = (held_bins == cell) & ((held_feat[:, SPRINT_DIM] > 0.5) == sprint_on)
        return held_feat[m]

    g_l1, g_l0 = group(7, True), group(7, False)   # late-loss, sprint on/off
    g_w1, g_w0 = group(6, True), group(6, False)   # late-win,  sprint on/off
    sizes = {'late_loss_spr1': len(g_l1), 'late_loss_spr0': len(g_l0),
             'late_win_spr1': len(g_w1), 'late_win_spr0': len(g_w0)}

    if min(sizes.values()) == 0:
        raise ValueError(f"empty group in interaction_gate: {sizes} -- "
                         f"cannot compute DiD, check held-out split")

    with torch.no_grad():
        m = lambda a: discriminator(torch.as_tensor(a, dtype=torch.float32)).mean().item()
        did = (m(g_l1) - m(g_l0)) - (m(g_w1) - m(g_w0))

    warning = None
    if min(sizes.values()) < min_group_size:
        warning = (f"group size below {min_group_size}: {sizes} -- "
                   f"this DiD estimate is noisy, consider train+held-out combined")

    return bool(did > threshold), did, sizes, warning


# ---------------------------------------------------------------------------
# D10 -- real-group hinge, added after GATE 1 showed all synthetic-hinge
# variants (D6-D9) have a large train/gate DiD gap (0.85-0.94 synthetic vs
# 0.16-0.26 real, D8 sign-flipped to -0.12). Diagnosis: the synthetic hinge
# lets the model learn a shortcut keyed on the literal (dim135, context)
# values against ONE reused base row's content, rather than a genuine
# content-context interaction. This trains the EXACT quantity the gate
# measures directly, on real minibatches from four disjoint real groups --
# no synthetic overwrite anywhere, so there is no literal-value shortcut
# to learn.
# ---------------------------------------------------------------------------

def build_real_groups(train_feat, train_bins):
    """Pre-split TRAIN data into the four real groups the gate/hinge both
    use. Call once; pass the returned dict into real_interaction_hinge
    every step."""
    def grp(cell, sprint_on):
        m = (train_bins == cell) & ((train_feat[:, SPRINT_DIM] > 0.5) == sprint_on)
        return train_feat[m]
    groups = {
        'late_loss_spr1': grp(7, True), 'late_loss_spr0': grp(7, False),
        'late_win_spr1': grp(6, True), 'late_win_spr0': grp(6, False),
    }
    sizes = {k: len(v) for k, v in groups.items()}
    if min(sizes.values()) < 50:
        print(f"  WARNING build_real_groups: thin group(s) {sizes}")
    return groups


def real_interaction_hinge(discriminator, groups, rng, batch_size=64,
                           margin=DEFAULT_INT_MARGIN):
    """Fully real, no synthesis. Draws batch_size (or fewer, if the group
    is smaller) REAL rows from each of the four groups every step and
    computes DiD on their mean discriminator output, WITH gradient.

    did = [D(late-loss,spr1) - D(late-loss,spr0)]
        - [D(late-win, spr1) - D(late-win, spr0)]

    Same target quantity as interaction_gate, computed on TRAIN split
    with gradients enabled instead of held-out with torch.no_grad().
    """
    def draw(key):
        pool = groups[key]
        n = min(batch_size, len(pool))
        idx = rng.choice(len(pool), size=n, replace=(n < batch_size))
        return torch.as_tensor(pool[idx], dtype=torch.float32)

    d_l1 = discriminator(draw('late_loss_spr1')).mean()
    d_l0 = discriminator(draw('late_loss_spr0')).mean()
    d_w1 = discriminator(draw('late_win_spr1')).mean()
    d_w0 = discriminator(draw('late_win_spr0')).mean()

    did = (d_l1 - d_l0) - (d_w1 - d_w0)
    hinge = torch.relu(margin - did)
    return hinge, did.detach()
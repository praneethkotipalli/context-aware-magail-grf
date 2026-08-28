"""
oracle_sticky_state.py

Before rebuilding the whole feature pipeline to add sticky_actions[8]:
check (1) whether it varies by context cell for humans (should, per
CSI_SAP=-31.6pp) while staying roughly flat for the agent (per the
toggle-rate check, which found the AGENT roughly context-independent),
and (2) what ceiling this signal alone would give on the actual gate
populations, using the SAME closed-form Bayes-optimal approach as
oracle_discriminator.py -- just swapping which raw field defines "sprint".

Reads raw .npz files directly (sticky_actions isn't in the cached
137-dim features at all). Reuses classify_bin, held_out_split (SAME
seed=0 split, so results are directly comparable to everything else),
and context_shift_scoring's LATE_WINNING/EARLY_LOSING population
definitions -- reimplemented at the raw-signal level since the existing
model-based harness expects a 137-dim vector, which sticky_actions isn't
part of yet.
"""

import os, glob
import numpy as np
from pathlib import Path

from build_expert_dataset import classify_bin
from context_balanced_sampler import CELL_NAMES, N_CELLS
from held_out_split import make_episode_split
from context_shift_scoring import LATE_WINNING, EARLY_LOSING

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
DEMO_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations", "episodes")
MAPPO_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "baseline_rollouts_for_verification")
SPRINT_STICKY_INDEX = 8


def per_step_records(episode_dir, episode_indices=None):
    """Returns arrays: cell, is_sprint_state, t_norm, delta_score -- one
    row per step, across the given episodes (or all, if None). episode_indices
    filters by the SAME enumeration order build_expert_dataset.py uses
    (sorted glob), so indices line up with make_episode_split's output."""
    paths = sorted(glob.glob(os.path.join(episode_dir, "*.npz")))
    if episode_indices is not None:
        paths = [paths[i] for i in episode_indices]

    cells, is_sprint, t_norms, delta_scores = [], [], [], []
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        n = len(d['steps_left'])
        steps_left = d['steps_left']
        score_left = d['score_left']
        score_right = d['score_right']
        sticky = d['sticky_actions']

        for t in range(n):
            t_norm = float(steps_left[t]) / 3001.0
            delta = int(score_left[t]) - int(score_right[t])
            cells.append(classify_bin(t_norm, delta))
            is_sprint.append(bool(sticky[t, SPRINT_STICKY_INDEX] == 1))
            t_norms.append(t_norm)
            delta_scores.append(delta)

    return (np.array(cells), np.array(is_sprint), np.array(t_norms), np.array(delta_scores))


def build_cell_rates(cells, is_sprint):
    rates = np.zeros(N_CELLS)
    counts = np.zeros(N_CELLS, dtype=int)
    for c in range(N_CELLS):
        mask = cells == c
        n_total = mask.sum()
        n_sprint = (mask & is_sprint).sum()
        counts[c] = n_total
        rates[c] = (n_sprint + 1.0) / (n_total + 2.0)
    return rates, counts


def oracle_shift(p_e, p_a, is_sprint):
    """log(p_e) - log(p_a) if sprinting, else log(1-p_e) - log(1-p_a).
    Returns the pre-sigmoid logit, matching the real oracle's convention."""
    if is_sprint:
        return np.log(p_e) - np.log(p_a)
    return np.log(1 - p_e) - np.log(1 - p_a)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run():
    print("Loading expert TRAIN split (same seed=0 episode split as everywhere else)...")
    expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
    train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)

    train_cells, train_sprint, _, _ = per_step_records(DEMO_DIR, episode_indices=train_ep.tolist())
    p_expert_state, expert_counts = build_cell_rates(train_cells, train_sprint)

    print("Loading full frozen-MAPPO set...")
    mappo_cells, mappo_sprint, _, _ = per_step_records(MAPPO_DIR)
    p_agent_state, agent_counts = build_cell_rates(mappo_cells, mappo_sprint)

    print(f"\n{'cell':12s} {'expert n':>9s} {'P(sprint-STATE|expert)':>23s} {'agent n':>9s} {'P(sprint-STATE|agent)':>22s}")
    for c in range(N_CELLS):
        print(f"  {CELL_NAMES[c]:10s} {expert_counts[c]:9d} {p_expert_state[c]*100:22.2f}% "
              f"{agent_counts[c]:9d} {p_agent_state[c]*100:21.2f}%")

    expert_spread = p_expert_state.max() - p_expert_state.min()
    agent_spread = p_agent_state.max() - p_agent_state.min()
    print(f"\n  expert sprint-STATE spread across cells: {expert_spread*100:.1f}pp "
          f"({'varies meaningfully -- context-conditional signal present' if expert_spread > 0.10 else 'roughly flat -- weaker signal than hoped'})")
    print(f"  agent sprint-STATE spread across cells:  {agent_spread*100:.1f}pp "
          f"({'roughly flat -- stays a usable near-marginal cue' if agent_spread < 0.10 else 'NOT flat -- more complex than assumed'})")

    print("\nLoading HELD-OUT episodes for the actual gate populations...")
    held_cells, held_sprint, held_t_norm, held_delta = per_step_records(DEMO_DIR, episode_indices=held_ep.tolist())

    lw_mask = (held_t_norm <= LATE_WINNING['t_norm_max']) & (held_delta > 0)
    el_mask = (held_t_norm >= EARLY_LOSING['t_norm_min']) & (held_delta < 0)

    LATE_WIN_CELL = classify_bin(0.1, 2)    # matches swap target used throughout
    EARLY_LOSS_CELL = classify_bin(0.85, -2)

    print(f"\nlate-winning population: n={lw_mask.sum()}  (swap target: early/loss cell)")
    shifts_lw = np.array([
        sigmoid(oracle_shift(p_expert_state[EARLY_LOSS_CELL], p_agent_state[EARLY_LOSS_CELL], s))
        - sigmoid(oracle_shift(p_expert_state[c], p_agent_state[c], s))
        for c, s in zip(held_cells[lw_mask], held_sprint[lw_mask])
    ])
    print(f"  mean|shift| = {np.abs(shifts_lw).mean():.4f}  (threshold 0.1)")
    print(f"  frac exceeding 0.1 = {(np.abs(shifts_lw) > 0.1).mean()*100:.1f}%")

    print(f"\nearly-losing population: n={el_mask.sum()}  (swap target: late/win cell)")
    shifts_el = np.array([
        sigmoid(oracle_shift(p_expert_state[LATE_WIN_CELL], p_agent_state[LATE_WIN_CELL], s))
        - sigmoid(oracle_shift(p_expert_state[c], p_agent_state[c], s))
        for c, s in zip(held_cells[el_mask], held_sprint[el_mask])
    ])
    print(f"  mean|shift| = {np.abs(shifts_el).mean():.4f}  (threshold 0.1)")
    print(f"  frac exceeding 0.1 = {(np.abs(shifts_el) > 0.1).mean()*100:.1f}%")

    both_pass = (np.abs(shifts_lw).mean() > 0.1) and (np.abs(shifts_el).mean() > 0.1)
    print("\n" + "=" * 70)
    print(f"VERDICT: sticky-state oracle {'PASSES' if both_pass else 'FAILS'} both directions")
    print("=" * 70)
    if both_pass:
        print("Adding sticky_actions[8] as a real input dimension is worth the")
        print("pipeline rebuild -- this signal alone clears the threshold.")
    else:
        print("Even the TRUE SAP signal doesn't clear 0.1 on its own here --")
        print("worth checking whether the marginal gap (agent~87% vs expert~18%,")
        print("roughly constant across cells) is swamping the CONDITIONAL")
        print("variation this test is trying to isolate, same shortcut risk as before.")


if __name__ == '__main__':
    run()
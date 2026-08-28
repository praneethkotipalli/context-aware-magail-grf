"""
verify_sap_definition.py

Tests whether the project's documented SAP statistics (human 9.9/19.9/20.9%
by win/draw/loss, baseline ~93.5%) match sticky_actions[8] (continuous
sprint STATE) rather than action_captured==13 (discrete toggle EVENT,
what feature_derivation.py's action one-hot actually encodes).

Reads raw .npz files directly -- does NOT go through feature_derivation.py,
since the point is checking a signal that pipeline never touches.
"""

import os, glob
import numpy as np
from pathlib import Path

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
DEMO_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations", "episodes")
MAPPO_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "baseline_rollouts_for_verification")

SPRINT_ACTION_INDEX = 13
SPRINT_STICKY_INDEX = 8


def analyze(episode_dir, label):
    paths = sorted(glob.glob(os.path.join(episode_dir, "*.npz")))
    print(f"\n{label}: {len(paths)} episodes")

    toggle_rates, state_rates = [], []
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        actions = d['action_captured']
        sticky = d['sticky_actions']  # shape (T, 10) presumably

        toggle_rate = np.mean(actions == SPRINT_ACTION_INDEX)
        state_rate = np.mean(sticky[:, SPRINT_STICKY_INDEX] == 1)

        toggle_rates.append(toggle_rate)
        state_rates.append(state_rate)

    toggle_rates = np.array(toggle_rates)
    state_rates = np.array(state_rates)

    print(f"  discrete TOGGLE rate (action_captured==13):  mean {toggle_rates.mean()*100:.2f}%  "
          f"(range {toggle_rates.min()*100:.2f}%-{toggle_rates.max()*100:.2f}%)")
    print(f"  continuous STATE rate (sticky_actions[8]):   mean {state_rates.mean()*100:.2f}%  "
          f"(range {state_rates.min()*100:.2f}%-{state_rates.max()*100:.2f}%)")
    return toggle_rates, state_rates


print("Testing: does sticky_actions[8] match the documented SAP figures")
print("(human 9.9/19.9/20.9% by outcome, baseline ~93.5%) better than")
print("action_captured==13 does?")

demo_toggle, demo_state = analyze(DEMO_DIR, "HUMAN DEMONSTRATIONS")
mappo_toggle, mappo_state = analyze(MAPPO_DIR, "FROZEN MAPPO BASELINE")

print("\n" + "=" * 70)
print("COMPARISON AGAINST DOCUMENTED SAP FIGURES")
print("=" * 70)
print(f"  documented human SAP (all outcomes, episode-level mean): ~17.7% (roughly)")
print(f"  documented baseline SAP: ~93.5%")
print()
print(f"  measured human STATE rate:   {demo_state.mean()*100:.2f}%  <- compare to ~17.7%")
print(f"  measured human TOGGLE rate:  {demo_toggle.mean()*100:.2f}%  <- compare to ~17.7%")
print()
print(f"  measured baseline STATE rate:  {mappo_state.mean()*100:.2f}%  <- compare to ~93.5%")
print(f"  measured baseline TOGGLE rate: {mappo_toggle.mean()*100:.2f}%  <- compare to ~93.5%")
print()
print("Whichever pair (STATE or TOGGLE) lands close to the documented")
print("figures tells us which signal your original SAP statistics actually")
print("used -- and therefore whether feature_derivation.py is missing it.")
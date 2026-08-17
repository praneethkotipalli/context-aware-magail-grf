"""
src/grf_baseline/inspect_sticky_actions.py

Confirms, empirically, which index in the sticky_actions vector corresponds
to sprint (and dribble, and direction stickies) by taking a KNOWN action and
observing which bit flips.

This must be run BEFORE finalising compute_sap() / compute_sbf() — do not
hard-code an assumed index from documentation. Confirm it here first.

Run with dissertation-grf environment activated:
    python src/grf_baseline/inspect_sticky_actions.py
"""

import numpy as np
import gfootball.env as football_env


# GRF core action indices (from official docs — used only to WORK/take actions
# during this test, not assumed for the sticky_actions vector itself)
ACTION_IDLE = 0
ACTION_SPRINT = 13
ACTION_RELEASE_SPRINT = 15
ACTION_DRIBBLE = 17
ACTION_RELEASE_DRIBBLE = 18


def make_raw_env():
    env = football_env.create_environment(
        env_name='5_vs_5',
        representation='raw',
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=False,
    )
    return env


def print_sticky_vector(obs, label):
    """Print sticky_actions for every controlled player."""
    print(f"\n--- {label} ---")
    for i, player_obs in enumerate(obs):
        sticky = player_obs['sticky_actions']
        print(f"  Player {i}: {sticky}  (len={len(sticky)})")


def diff_sticky_vectors(before, after, label):
    """Print which indices changed between two sticky_actions readings."""
    print(f"\n--- Bit changes: {label} ---")
    for i in range(len(before)):
        b = before[i]['sticky_actions']
        a = after[i]['sticky_actions']
        b_arr = np.array(b)
        a_arr = np.array(a)
        changed = np.where(b_arr != a_arr)[0]
        if len(changed) > 0:
            print(f"  Player {i}: indices changed = {list(changed)}  "
                  f"(before={b_arr[changed]}, after={a_arr[changed]})")
        else:
            print(f"  Player {i}: no change")


def build_action_array(n_agents, active_agent_idx, action_for_active, other_action=ACTION_IDLE):
    """Build an action array where only one agent takes a specific action, rest idle."""
    actions = [other_action] * n_agents
    actions[active_agent_idx] = action_for_active
    return actions


def main():
    print("=" * 70)
    print("STICKY ACTION VECTOR INSPECTION (raw representation)")
    print("=" * 70)

    env = make_raw_env()
    obs = env.reset()

    n_agents = len(obs)
    print(f"\nNumber of controlled agents: {n_agents}")
    print(f"Observation type per agent: {type(obs[0])}")

    if isinstance(obs[0], dict):
        print(f"Keys in player observation: {list(obs[0].keys())}")
    else:
        print("[ERROR] Raw observation is not a dict as expected. "
              "Check gfootball version — structure may differ.")
        env.close()
        return

    # ── Step 0: baseline sticky vector (should be all zeros right after reset) ──
    print_sticky_vector(obs, "STEP 0 — Immediately after reset() (expect all zeros)")
    sticky_len = len(obs[0]['sticky_actions'])
    print(f"\nSticky action vector length: {sticky_len}")

    # ── Test 1: SPRINT ON ──
    # All agents idle except agent 0, who takes SPRINT
    actions = build_action_array(n_agents, active_agent_idx=0, action_for_active=ACTION_SPRINT)
    print(f"\n[ACTION TAKEN] agent 0 = SPRINT (idx {ACTION_SPRINT}), others = IDLE")
    obs_before_sprint = obs
    obs, reward, done, info = env.step(actions)
    print_sticky_vector(obs, "STEP 1 — After taking SPRINT action")
    diff_sticky_vectors(obs_before_sprint, obs, "reset -> after SPRINT")

    # ── Test 2: Hold — take IDLE again, sprint should PERSIST (sticky!) ──
    actions = build_action_array(n_agents, active_agent_idx=0, action_for_active=ACTION_IDLE)
    print(f"\n[ACTION TAKEN] agent 0 = IDLE (sprint should still be ACTIVE — that's the 'sticky' part)")
    obs_before_idle = obs
    obs, reward, done, info = env.step(actions)
    print_sticky_vector(obs, "STEP 2 — After taking IDLE (sprint bit should PERSIST as 1)")
    diff_sticky_vectors(obs_before_idle, obs, "after SPRINT -> after IDLE (expect NO change if truly sticky)")

    # ── Test 3: RELEASE_SPRINT — bit should flip back to 0 ──
    actions = build_action_array(n_agents, active_agent_idx=0, action_for_active=ACTION_RELEASE_SPRINT)
    print(f"\n[ACTION TAKEN] agent 0 = RELEASE_SPRINT (idx {ACTION_RELEASE_SPRINT})")
    obs_before_release = obs
    obs, reward, done, info = env.step(actions)
    print_sticky_vector(obs, "STEP 3 — After RELEASE_SPRINT (sprint bit should flip back to 0)")
    diff_sticky_vectors(obs_before_release, obs, "after IDLE-hold -> after RELEASE_SPRINT")

    # ── Test 4: DRIBBLE ON (to also confirm the dribble bit, useful later) ──
    actions = build_action_array(n_agents, active_agent_idx=0, action_for_active=ACTION_DRIBBLE)
    print(f"\n[ACTION TAKEN] agent 0 = DRIBBLE (idx {ACTION_DRIBBLE})")
    obs_before_dribble = obs
    obs, reward, done, info = env.step(actions)
    print_sticky_vector(obs, "STEP 4 — After DRIBBLE action")
    diff_sticky_vectors(obs_before_dribble, obs, "after RELEASE_SPRINT -> after DRIBBLE")

    # ── Test 5: RELEASE_DRIBBLE ──
    actions = build_action_array(n_agents, active_agent_idx=0, action_for_active=ACTION_RELEASE_DRIBBLE)
    print(f"\n[ACTION TAKEN] agent 0 = RELEASE_DRIBBLE (idx {ACTION_RELEASE_DRIBBLE})")
    obs_before_rdribble = obs
    obs, reward, done, info = env.step(actions)
    print_sticky_vector(obs, "STEP 5 — After RELEASE_DRIBBLE")
    diff_sticky_vectors(obs_before_rdribble, obs, "after DRIBBLE -> after RELEASE_DRIBBLE")

    env.close()

    print("\n" + "=" * 70)
    print("HOW TO READ THIS OUTPUT")
    print("=" * 70)
    print("""
Look at the "Bit changes" sections for Player 0 only (the other players
were held idle throughout and should show no change at any step).

1. STEP 0 -> STEP 1 (after SPRINT): exactly one index should flip 0 -> 1.
   THAT INDEX IS YOUR SPRINT BIT. Write it down.

2. STEP 1 -> STEP 2 (SPRINT held, then IDLE taken): the sprint bit should
   show "no change" -- this is what confirms it is a STICKY action (it
   persists across steps without needing the action repeated).
   If the bit reset to 0 here, sticky behaviour is not what you assumed --
   re-examine before finalising your metric code.

3. STEP 2 -> STEP 3 (after RELEASE_SPRINT): the same index should flip
   1 -> 0. This confirms both directions of the toggle.

4. STEP 3 -> STEP 4 (after DRIBBLE): a DIFFERENT index should flip 0 -> 1.
   This is your dribble bit (not needed for SAP/SBF, but useful to know
   for future metrics or sanity checks).

5. STEP 4 -> STEP 5 (after RELEASE_DRIBBLE): that same dribble index
   should flip back to 0.

ACTION REQUIRED AFTER RUNNING THIS:
Update src/metrics/calculators.py:

    SPRINT_STICKY_INDEX = <the index you confirmed above>

And change compute_sap / compute_sbf to operate on a logged sticky-action
time series (extracted via obs[i]['sticky_actions'][SPRINT_STICKY_INDEX]
at every step for every agent) instead of the discrete action array.
""")


if __name__ == '__main__':
    main()
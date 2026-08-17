import sys, os, csv, json, argparse
import numpy as np
import pygame
from pathlib import Path

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")

sys.path.insert(0, os.path.join(DISS_ROOT, "football"))
sys.path.insert(0, os.path.join(DISS_ROOT, "GRF_MARL"))
sys.path.insert(0, os.path.join(DISS_ROOT, "context-aware-magail-grf", "src", "grf_baseline"))
sys.path.insert(0, os.path.join(DISS_ROOT, "context-aware-magail-grf", "src", "metrics"))
sys.path.insert(0, os.path.join(DISS_ROOT, "GRF_MARL", "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"))

from gfootball.env import config
from gfootball.env import football_env
from calculator import compute_sap, compute_mecha

OUT_ROOT = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations")
EPISODES_DIR = os.path.join(OUT_ROOT, "episodes")      # one .npz per episode
AUDIT_DIR = os.path.join(OUT_ROOT, "audit")             # one .json per episode
SUMMARY_CSV = os.path.join(OUT_ROOT, "episode_summary.csv")  # one row per episode, all episodes

REGIONS = ['early_neutral', 'mid_neutral', 'late_winning', 'late_losing', 'other']

# --- discrete action ids (GRF default 19-action set) ---
ACTION_IDLE, ACTION_LEFT, ACTION_TOP_LEFT, ACTION_TOP, ACTION_TOP_RIGHT = 0, 1, 2, 3, 4
ACTION_RIGHT, ACTION_BOTTOM_RIGHT, ACTION_BOTTOM, ACTION_BOTTOM_LEFT = 5, 6, 7, 8
ACTION_LONG_PASS, ACTION_HIGH_PASS, ACTION_SHORT_PASS, ACTION_SHOT = 9, 10, 11, 12
ACTION_SPRINT, ACTION_RELEASE_DIRECTION, ACTION_RELEASE_SPRINT = 13, 14, 15
ACTION_SLIDING, ACTION_DRIBBLE, ACTION_RELEASE_DRIBBLE = 16, 17, 18

ACTION_NAMES = {
    0: 'IDLE', 1: 'LEFT', 2: 'TOP_LEFT', 3: 'TOP', 4: 'TOP_RIGHT',
    5: 'RIGHT', 6: 'BOTTOM_RIGHT', 7: 'BOTTOM', 8: 'BOTTOM_LEFT',
    9: 'LONG_PASS', 10: 'HIGH_PASS', 11: 'SHORT_PASS', 12: 'SHOT',
    13: 'SPRINT', 14: 'RELEASE_DIRECTION', 15: 'RELEASE_SPRINT',
    16: 'SLIDING', 17: 'DRIBBLE', 18: 'RELEASE_DRIBBLE',
}

DIR_ORDER = [ACTION_LEFT, ACTION_TOP_LEFT, ACTION_TOP, ACTION_TOP_RIGHT,
             ACTION_RIGHT, ACTION_BOTTOM_RIGHT, ACTION_BOTTOM, ACTION_BOTTOM_LEFT]


# --- verified action capture (confirmed clean: all 19 actions, sprint 17/17, dribble 2/2) ---
class ActionCapture:
    def __init__(self):
        self._prev_keys = None

    def _direction(self, keys):
        up, down = keys[pygame.K_UP], keys[pygame.K_DOWN]
        left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        if up and right: return ACTION_TOP_RIGHT
        if up and left: return ACTION_TOP_LEFT
        if down and right: return ACTION_BOTTOM_RIGHT
        if down and left: return ACTION_BOTTOM_LEFT
        if up: return ACTION_TOP
        if down: return ACTION_BOTTOM
        if left: return ACTION_LEFT
        if right: return ACTION_RIGHT
        return None

    def capture(self, sticky_before, sticky_after, has_possession):
        dir_before, dir_after = sticky_before[:8], sticky_after[:8]
        sprint_before, sprint_after = sticky_before[8], sticky_after[8]
        dribble_before, dribble_after = sticky_before[9], sticky_after[9]

        idx_before = int(np.argmax(dir_before)) if any(dir_before) else None
        idx_after = int(np.argmax(dir_after)) if any(dir_after) else None

        keys = pygame.key.get_pressed()
        prev = self._prev_keys

        def rising(k):
            return keys[k] and (prev is None or not prev[k])
        computed_dir = self._direction(keys)


        

        if sprint_before != sprint_after:
            action = ACTION_SPRINT if sprint_after == 1 else ACTION_RELEASE_SPRINT
        elif dribble_before != dribble_after:
            action = ACTION_DRIBBLE if dribble_after == 1 else ACTION_RELEASE_DRIBBLE
        elif has_possession and rising(pygame.K_w):
            action = ACTION_LONG_PASS
        elif has_possession and rising(pygame.K_a):
            action = ACTION_HIGH_PASS
        elif has_possession and rising(pygame.K_s):
            action = ACTION_SHORT_PASS
        elif has_possession and rising(pygame.K_d):
            action = ACTION_SHOT
        elif (not has_possession) and rising(pygame.K_a):
            action = ACTION_SLIDING
        elif idx_after is not None and idx_after != idx_before:
            action = DIR_ORDER[idx_after]
        elif idx_before is not None and idx_after is None:
            action = ACTION_RELEASE_DIRECTION
        elif idx_after is not None and idx_after == idx_before:
            action = computed_dir if computed_dir == DIR_ORDER[idx_after] else ACTION_IDLE
        else:
            action = ACTION_IDLE

        self._prev_keys = keys
        return action


def classify_region(t_norm, delta_score):
    """Matches the locked methodology's four match regions (Section 3.2.2)."""
    if t_norm < 0.3 and delta_score == 0:
        return 'early_neutral'
    if 0.4 <= t_norm <= 0.6 and delta_score == 0:
        return 'mid_neutral'
    if t_norm > 0.7 and delta_score > 0:
        return 'late_winning'
    if t_norm > 0.7 and delta_score < 0:
        return 'late_losing'
    return 'other'


def make_env(level):
    cfg = config.Config({'action_set': 'default', 'players': ['keyboard:left_players=1'],
                          'level': level, 'representation': 'raw', 'real_time': True,
                          'rewards': 'scoring'})
    env = football_env.FootballEnv(cfg)
    env.render()
    return env


def next_id():
    if not os.path.exists(SUMMARY_CSV):
        return 1
    with open(SUMMARY_CSV) as f:
        rows = list(csv.DictReader(f))
    return (max(int(r['episode_id']) for r in rows) + 1) if rows else 1


def record_episode(env, ctx, ep_id, capture):
    obs = env.reset()

    steps_left, score_left, score_right = [], [], []
    left_team, left_team_dir, right_team, right_team_dir = [], [], [], []
    ball, ball_dir = [], []
    ball_owned_team, ball_owned_player = [], []
    sticky_actions, action_captured, has_possession, context_region = [], [], [], []

    region_counts = {r: 0 for r in REGIONS}
    action_counts = {i: 0 for i in range(19)}
    step, done = 0, False

    while not done:
        sticky_before = np.array(obs['left_agent_sticky_actions'][0])
        possession = bool(obs['ball_owned_team'] == 0)
        t_norm = obs['steps_left'] / 3001.0
        delta_score = obs['score'][0] - obs['score'][1]
        region = classify_region(t_norm, delta_score)
        region_counts[region] += 1

        # log this step's state BEFORE stepping
        steps_left.append(obs['steps_left'])
        score_left.append(obs['score'][0]); score_right.append(obs['score'][1])
        left_team.append(obs['left_team']); left_team_dir.append(obs['left_team_direction'])
        right_team.append(obs['right_team']); right_team_dir.append(obs['right_team_direction'])
        ball.append(obs['ball']); ball_dir.append(obs['ball_direction'])
        ball_owned_team.append(obs['ball_owned_team']); ball_owned_player.append(obs['ball_owned_player'])
        sticky_actions.append(sticky_before)
        has_possession.append(possession)
        context_region.append(region)

        next_obs, reward, done, info = env.step([])
        sticky_after = np.array(next_obs['left_agent_sticky_actions'][0])

        action = capture.capture(sticky_before, sticky_after, possession)
        action_captured.append(action)
        action_counts[action] += 1

        obs = next_obs
        step += 1

    n_steps = step
    region_arr = np.array(context_region, dtype='<U16')

    # ---- 1. per-episode raw data: .npz ----
    os.makedirs(EPISODES_DIR, exist_ok=True)
    npz_path = os.path.join(EPISODES_DIR, f"{ctx}_ep{ep_id}.npz")
    np.savez_compressed(
        npz_path,
        episode_id=ep_id, context_label=ctx,
        steps_left=np.array(steps_left),
        score_left=np.array(score_left), score_right=np.array(score_right),
        left_team=np.array(left_team), left_team_direction=np.array(left_team_dir),
        right_team=np.array(right_team), right_team_direction=np.array(right_team_dir),
        ball=np.array(ball), ball_direction=np.array(ball_dir),
        ball_owned_team=np.array(ball_owned_team), ball_owned_player=np.array(ball_owned_player),
        sticky_actions=np.array(sticky_actions), action_captured=np.array(action_captured),
        has_possession=np.array(has_possession), context_region=region_arr,
    )

    # ---- metrics, computed from the same arrays just saved ----
    sprint_arr = np.array(sticky_actions)[:, 8:9]
    pos_arr = np.array(left_team)[:, :5]
    poss_arr = np.array(has_possession)
    sap = compute_sap(sprint_arr)
    mecha = compute_mecha(pos_arr, poss_arr)
    possession_fraction = float(poss_arr.mean())
    sprint_on, sprint_off = action_counts[13], action_counts[15]
    dribble_on, dribble_off = action_counts[17], action_counts[18]

    # ---- 2. per-episode audit file: .json, human-readable ----
    os.makedirs(AUDIT_DIR, exist_ok=True)
    audit_path = os.path.join(AUDIT_DIR, f"{ctx}_ep{ep_id}_audit.json")
    audit = {
        'episode_id': ep_id, 'context_label': ctx, 'n_steps': n_steps,
        'sap': round(sap, 2), 'mecha': round(mecha, 4),
        'possession_fraction': round(possession_fraction, 3),
        'region_counts': region_counts,
        'action_counts': {ACTION_NAMES[i]: c for i, c in action_counts.items()},
        'sprint_on_off': [sprint_on, sprint_off],
        'dribble_on_off': [dribble_on, dribble_off],
        'flags': [],
    }
    if abs(sprint_on - sprint_off) > 1:
        audit['flags'].append('sprint on/off mismatch > 1 -- check capture logic')
    if abs(dribble_on - dribble_off) > 1:
        audit['flags'].append('dribble on/off mismatch > 1 -- check capture logic')
    if action_counts[0] / n_steps > 0.5:
        audit['flags'].append('over 50% IDLE -- check keyboard focus / capture logic')
    with open(audit_path, 'w') as af:
        json.dump(audit, af, indent=2)

    # ---- 3. all-episodes summary: single CSV, one row appended per episode ----
    write_header = not os.path.exists(SUMMARY_CSV) or os.path.getsize(SUMMARY_CSV) == 0
    with open(SUMMARY_CSV, 'a', newline='') as sf:
        w = csv.writer(sf)
        if write_header:
            w.writerow(['episode_id', 'context_label', 'n_steps', 'sap', 'mecha',
                        'possession_fraction', 'region_counts', 'npz_path', 'audit_path', 'flags'])
        w.writerow([ep_id, ctx, n_steps, round(sap, 2), round(mecha, 4),
                    round(possession_fraction, 3), json.dumps(region_counts),
                    npz_path, audit_path, '; '.join(audit['flags'])])

    print(f"[{ctx}] ep{ep_id} steps={n_steps} SAP={sap:.1f}% MECHA={mecha:.4f} possession={possession_fraction:.2f}")
    if audit['flags']:
        print(f"  FLAGS: {audit['flags']}")
    print(f"  region coverage this episode: {region_counts}")
    print_running_coverage()


def print_running_coverage():
    """Live tally across ALL episodes recorded so far."""
    if not os.path.exists(SUMMARY_CSV):
        return
    totals = {r: 0 for r in REGIONS}
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            try:
                counts = json.loads(row['region_counts'])
                for r in REGIONS:
                    totals[r] += counts.get(r, 0)
            except Exception:
                continue
    print("  RUNNING TOTAL across all episodes:")
    for r in ['early_neutral', 'mid_neutral', 'late_winning', 'late_losing']:
        flag = "OK" if totals[r] > 0 else "MISSING -- keep playing to hit this region"
        print(f"    {r:15s}: {totals[r]:5d} steps  [{flag}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', default='early_neutral')
    p.add_argument('--level', default='5_vs_5_d06')
    p.add_argument('--n_episodes', type=int, default=1)
    args = p.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    env = make_env(args.level)
    capture = ActionCapture()
    sid = next_id()
    try:
        for i in range(args.n_episodes):
            record_episode(env, args.context, sid + i, capture)
    except KeyboardInterrupt:
        print("Stopped; completed episodes saved.")
    finally:
        env.close()


if __name__ == '__main__':
    main()
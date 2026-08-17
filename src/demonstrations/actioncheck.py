# action_checker.py -- run this alone, writes nothing, just verifies capture is correct.
import sys, os
import pygame
from pathlib import Path

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
sys.path.insert(0, os.path.join(DISS_ROOT, "football"))
sys.path.insert(0, os.path.join(DISS_ROOT, "GRF_MARL"))

from gfootball.env import config
from gfootball.env import football_env

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


import numpy as np

DIR_ORDER = [ACTION_LEFT, ACTION_TOP_LEFT, ACTION_TOP, ACTION_TOP_RIGHT,
             ACTION_RIGHT, ACTION_BOTTOM_RIGHT, ACTION_BOTTOM, ACTION_BOTTOM_LEFT]

class ActionCapture:
    def __init__(self):
        self._prev_keys = None
        self._dir_order_warned = False

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

        # cross-check the assumed sticky_actions direction ordering the first
        # time we see a fresh direction engage -- flag once if it disagrees
        if idx_after is not None and idx_after != idx_before:
            expected = DIR_ORDER[idx_after]
            if computed_dir is not None and computed_dir != expected:
                print(f"    MISMATCH: sticky idx {idx_after} implies {ACTION_NAMES[expected]}, "f"keys suggest {ACTION_NAMES[computed_dir]}")

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
            action = DIR_ORDER[idx_after]                 # new direction just engaged
        elif idx_before is not None and idx_after is None:
            action = ACTION_RELEASE_DIRECTION              # was moving, just stopped
        elif idx_after is not None and idx_after == idx_before:
            action = computed_dir if computed_dir == DIR_ORDER[idx_after] else ACTION_IDLE  # still held
        else:
            action = ACTION_IDLE

        self._prev_keys = keys
        return action

def main():
    cfg = config.Config({'action_set': 'default', 'players': ['keyboard:left_players=1'],
                          'level': '5_vs_5_d06', 'representation': 'raw', 'real_time': True,
                          'rewards': 'scoring'})
    env = football_env.FootballEnv(cfg)
    env.render()

    capture = ActionCapture()
    counts = {i: 0 for i in range(19)}
    last_logged = None

    obs = env.reset()
    step, done = 0, False
    print("Playing -- deliberately hit every key at least once: all 8 directions,")
    print("sprint on+off, dribble on+off, a pass while attacking, a shot, a slide")
    print("while defending. Ctrl+C when done to see the histogram.\n")

    try:
        while not done:
            sticky_before = obs['left_agent_sticky_actions'][0]
            has_possession = bool(obs['ball_owned_team'] == 0)

            next_obs, reward, done, info = env.step([])

            sticky_after = next_obs['left_agent_sticky_actions'][0]
            action = capture.capture(sticky_before, sticky_after, has_possession)
            counts[action] += 1

            if action != last_logged and action not in (ACTION_IDLE,):
                print(f"  step {step:4d}: {ACTION_NAMES[action]}")
                last_logged = action

            obs = next_obs
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    print("\nACTION HISTOGRAM:")
    for i in range(19):
        flag = "" if counts[i] > 0 else "  <- NEVER CAPTURED"
        print(f"  {i:2d} {ACTION_NAMES[i]:18s}: {counts[i]:5d}{flag}")

    sprint_on, sprint_off = counts[13], counts[15]
    dribble_on, dribble_off = counts[17], counts[18]
    print(f"\nsprint on/off: {sprint_on}/{sprint_off}  (should differ by at most 1)")
    print(f"dribble on/off: {dribble_on}/{dribble_off}  (should differ by at most 1)")


if __name__ == '__main__':
    main()
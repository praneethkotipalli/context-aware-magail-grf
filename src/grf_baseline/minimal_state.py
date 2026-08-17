"""
src/grf_baseline/minimal_state.py

Self-contained replacement for light_malib's State class, providing only what
enhanced_LightActionMask_5.FeatureEncoder.encode_each() actually needs.
"""

import numpy as np


class MinimalState:
    def __init__(self, n_player=5):
        self.num_player = n_player
        self.action_list = []
        self.last_loffside = np.zeros(self.num_player, np.float32)
        self.last_roffside = np.zeros(self.num_player, np.float32)
        self._current_obs = None

    def set_obs(self, obs):
        self._current_obs = obs

    @property
    def obs(self):
        return self._current_obs

    def get_offside(self, obs):
        ball = np.array(obs["ball"][:2])
        ally = np.array(obs["left_team"])
        enemy = np.array(obs["right_team"])

        if obs["game_mode"] != 0:
            self.last_loffside = np.zeros(self.num_player, np.float32)
            self.last_roffside = np.zeros(self.num_player, np.float32)
            return np.zeros(self.num_player, np.float32), np.zeros(self.num_player, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        if obs["ball_owned_team"] > -1:
            effective_ownball_team = obs["ball_owned_team"]
            effective_ownball_player = obs["ball_owned_player"]
            need_recalc = True
        else:
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(self.num_player, np.float32)
        right_offside = np.zeros(self.num_player, np.float32)

        if effective_ownball_team == 0:
            right_xs = np.array([obs["right_team"][k][0] for k in range(0, self.num_player)])
            right_xs.sort()
            offside_line = max(right_xs[-2], ball[0])
            for k in range(1, self.num_player):
                if (obs["left_team"][k][0] > offside_line
                        and k != effective_ownball_player
                        and obs["left_team"][k][0] > 0.0):
                    left_offside[k] = 1.0
        else:
            left_xs = np.array([obs["left_team"][k][0] for k in range(0, 5)])
            left_xs.sort()
            offside_line = min(left_xs[1], ball[0])
            for k in range(1, self.num_player):
                if (obs["right_team"][k][0] < offside_line
                        and k != effective_ownball_player
                        and obs["right_team"][k][0] < 0.0):
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside
        return left_offside, right_offside

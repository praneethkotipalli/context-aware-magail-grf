"""
src/grf_baseline/record_random_policy_rollout.py

Phase A data source (Section 3.3.3): 10,000 supervised discriminator
pre-training steps run against random-policy rollouts, before continuing
against frozen-MAPPO. Structurally identical to record_baseline_rollout.py
-- same raw .npz schema, same reference-player logic -- with the entire
actor/encoder block replaced by env.action_space.sample() (confirmed via
check_action_space.py: MultiDiscrete([19,19,19,19]), directly usable in
step(), no per-agent looping needed).

No actor checkpoint, no encoder, no MinimalState needed -- this has no
model dependency at all, only GRF itself.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")

sys.path.insert(0, GRF_MARL_ROOT)

import gfootball.env as football_env

N_EPISODES = 5  # placeholder -- real count decided once we see cell coverage
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "random_policy_rollouts")


def pick_reference_agent(raw_obs, agent_indices):
    """Identical logic to record_baseline_rollout.py -- nearest-to-ball
    among the 4 controlled agents, confirmed correct via
    check_active_player_mapping.py ([1,2,3,4])."""
    left_team = np.asarray(raw_obs[0]['left_team'])
    ball_xy = np.asarray(raw_obs[0]['ball'][:2])

    best_idx, best_dist = None, np.inf
    for i in agent_indices:
        player_idx = i + 1
        dist = np.linalg.norm(left_team[player_idx] - ball_xy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_idx + 1


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    env = football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=False,
    )

    for ep in range(N_EPISODES):
        raw_obs = env.reset()

        steps_left_log, score_left_log, score_right_log = [], [], []
        left_team_log, left_team_dir_log = [], []
        right_team_log, right_team_dir_log = [], []
        ball_log, ball_dir_log = [], []
        ball_owned_team_log, ball_owned_player_log = [], []
        sticky_actions_log, action_captured_log, has_possession_log = [], [], []

        step = 0
        done = False
        while not done and step < 3000:
            actions = env.action_space.sample()  # the only real change vs. record_baseline_rollout.py

            o = raw_obs[0]
            ref_agent_i, ref_player_i = pick_reference_agent(raw_obs, range(len(raw_obs)))

            steps_left_log.append(o['steps_left'])
            score_left_log.append(o['score'][0])
            score_right_log.append(o['score'][1])
            left_team_log.append(o['left_team'])
            left_team_dir_log.append(o['left_team_direction'])
            right_team_log.append(o['right_team'])
            right_team_dir_log.append(o['right_team_direction'])
            ball_log.append(o['ball'])
            ball_dir_log.append(o['ball_direction'])
            ball_owned_team_log.append(o['ball_owned_team'])
            ball_owned_player_log.append(o['ball_owned_player'])

            sticky_actions_log.append(raw_obs[ref_agent_i]['sticky_actions'])
            action_captured_log.append(int(actions[ref_agent_i]))
            has_possession_log.append(bool(o['ball_owned_team'] == 0))

            raw_obs, reward, done, info = env.step(list(actions))
            step += 1

        npz_path = os.path.join(OUT_DIR, f"random_policy_ep{ep}.npz")
        np.savez_compressed(
            npz_path,
            episode_id=ep, context_label='random_policy_rollout',
            steps_left=np.array(steps_left_log),
            score_left=np.array(score_left_log), score_right=np.array(score_right_log),
            left_team=np.array(left_team_log), left_team_direction=np.array(left_team_dir_log),
            right_team=np.array(right_team_log), right_team_direction=np.array(right_team_dir_log),
            ball=np.array(ball_log), ball_direction=np.array(ball_dir_log),
            ball_owned_team=np.array(ball_owned_team_log), ball_owned_player=np.array(ball_owned_player_log),
            sticky_actions=np.array(sticky_actions_log), action_captured=np.array(action_captured_log),
            has_possession=np.array(has_possession_log),
            context_region=np.full(len(steps_left_log), 'unclassified', dtype='<U16'),
        )
        print(f"ep{ep}: {step} steps -> {npz_path}")

    env.close()
    print(f"\nDone. {N_EPISODES} random-policy episodes saved to {OUT_DIR}")


if __name__ == '__main__':
    run()
"""
src/grf_baseline/record_baseline_rollout.py

Records a small number of FROZEN MAPPO baseline rollouts, saved in the
exact same raw-field schema as the human demonstration .npz files
(record_session.py), so the discriminator's feature_derivation.py /
feature_normalization.py pipeline can be verified against agent data,
not just human demonstration data.

NOT for real training data collection -- this is a verification tool only.
Run on Blackwell/WSL (GRF required); transfer the output .npz files to
Mac afterward for batch_verify_features.py.
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = "/home/u5749464/dissertation/context-aware-magail-grf"
GRF_MARL_ROOT = "/home/u5749464/dissertation/GRF_MARL"

sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.join(
    GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"
))

import gfootball.env as football_env
from minimal_state import MinimalState
from enhanced_LightActionMask_5 import FeatureEncoder

POLICY_NAME = "PassingMain_v2"
N_EPISODES = 5
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "baseline_rollouts_for_verification")


def pick_reference_agent(raw_obs, agent_indices):
    """Same 'nearest-to-ball' reference-player logic already used elsewhere
    in this project (ball-relation block, asymmetry handling). Here it
    picks WHICH of the 4 controlled agents to treat as the single
    'active'-equivalent player for this step's sticky_actions/action_captured
    fields -- matching the single-reference-player schema your demonstration
    .npz files use, so both sides of GAIL are constructed the same way."""
    left_team = np.asarray(raw_obs[0]['left_team'])  # global state, same across all 4 dicts
    ball_xy = np.asarray(raw_obs[0]['ball'][:2])

    best_idx, best_dist = None, np.inf
    for i in agent_indices:
        # NOTE: this assumes raw_obs[i] corresponds to left_team player (i+1)
        # (i.e. agent 0 -> player 1, agent 1 -> player 2, ..., since player 0
        # is the auto-played GK and not in this list). THIS IS UNVERIFIED --
        # print raw_obs[i] once and check for a field like 'active' or
        # 'designated_player' to confirm this mapping before trusting it.
        player_idx = i + 1
        dist = np.linalg.norm(left_team[player_idx] - ball_xy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_idx + 1  # (agent list index, left_team player index)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    actor_path = os.path.join(
        GRF_MARL_ROOT, "light_malib", "trained_models", "gr_football",
        "5_vs_5", POLICY_NAME, "actor.pt"
    )
    actor = torch.load(actor_path, map_location="cpu")
    actor.eval()
    encoder = FeatureEncoder()

    env = football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=False,
    )

    for ep in range(N_EPISODES):
        raw_obs = env.reset()

        # --- one-time schema check, first episode only ---
        if ep == 0:
            print("=== raw_obs[0] keys ===")
            print(list(raw_obs[0].keys()))
            print("=== does raw_obs[0] contain an 'active' or 'designated_player' field? ===")
            for k in ['active', 'designated_player']:
                if k in raw_obs[0]:
                    print(f"  {k}: {raw_obs[0][k]}  <- USE THIS to verify the agent-to-player mapping")
            print()

        steps_left_log, score_left_log, score_right_log = [], [], []
        left_team_log, left_team_dir_log = [], []
        right_team_log, right_team_dir_log = [], []
        ball_log, ball_dir_log = [], []
        ball_owned_team_log, ball_owned_player_log = [], []
        sticky_actions_log, action_captured_log, has_possession_log = [], [], []

        step = 0
        done = False
        while not done and step < 3000:
            actions = []
            for agent_idx in range(len(raw_obs)):
                state = MinimalState(n_player=5)
                state.set_obs(raw_obs[agent_idx])
                feat = encoder.encode_each(state)
                feat_tensor = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
                avail = encoder.get_available_actions(raw_obs[agent_idx], 0.0, [])
                mask_tensor = torch.as_tensor(avail, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    act, _, _, _ = actor(feat_tensor, None, None, mask_tensor, False, None)
                actions.append(int(act.item()))

            o = raw_obs[0]  # shared/global fields -- same across all 4 agent dicts
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

            # reference-player sticky_actions and action -- NOTE schema
            # difference from demo side: this env exposes 'sticky_actions'
            # directly per agent dict, not nested under 'left_agent_sticky_actions'
            sticky_actions_log.append(raw_obs[ref_agent_i]['sticky_actions'])
            action_captured_log.append(actions[ref_agent_i])  # the ACTUAL sampled action --
                                                                 # no reconstruction needed,
                                                                 # unlike the human keyboard side
            has_possession_log.append(bool(o['ball_owned_team'] == 0))

            raw_obs, reward, done, info = env.step(actions)
            step += 1

        npz_path = os.path.join(OUT_DIR, f"baseline_{POLICY_NAME}_ep{ep}.npz")
        np.savez_compressed(
            npz_path,
            episode_id=ep, context_label='baseline_rollout',
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
    print(f"\nDone. {N_EPISODES} baseline rollout episodes saved to {OUT_DIR}")
    print("Transfer this directory to Mac, then run batch_verify_features.py against it")
    print("(point EPISODES_DIR at this folder instead of data/demonstrations/episodes).")


if __name__ == '__main__':
    run()
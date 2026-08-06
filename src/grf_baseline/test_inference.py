import sys
sys.path.insert(0, "/home/u5749464/dissertation/GRF_MARL")
sys.path.insert(0, "/home/u5749464/dissertation/context-aware-magail-grf/src/grf_baseline")
sys.path.insert(0, "/home/u5749464/dissertation/GRF_MARL/light_malib/model/gr_football/enhanced_LightActionMask_5")

import torch
import numpy as np
from minimal_state import MinimalState
from enhanced_LightActionMask_5 import FeatureEncoder
import gfootball.env as football_env

actor = torch.load(
    "/home/u5749464/dissertation/GRF_MARL/light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt",
    map_location="cpu"
)
actor.eval()

env = football_env.create_environment(
    env_name='5_vs_5', representation='raw',
    number_of_left_players_agent_controls=4, render=False
)
raw_obs = env.reset()

encoder = FeatureEncoder()
state = MinimalState(n_player=5)
state.set_obs(raw_obs[0])

feat = encoder.encode_each(state)
feat_tensor = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
print("Input tensor shape:", feat_tensor.shape)

avail_actions = encoder.get_available_actions(raw_obs[0], 0.0, [])
action_mask_tensor = torch.as_tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
print("Action mask:", avail_actions)

with torch.no_grad():
    actions, rnn_states, log_probs, entropy = actor(
        feat_tensor, None, None, action_mask_tensor, False, None
    )

print("\nOUTPUT:")
print("Chosen action:", actions.item())
print("Log prob:", log_probs.item())

env.close()
print("\nPASS: real inference through the trained network succeeded")

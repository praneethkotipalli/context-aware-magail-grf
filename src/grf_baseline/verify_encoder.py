import sys
import os

PROJECT_ROOT = "/home/praneeth/dissertation/context-aware-magail-grf"
GRF_MARL_ROOT = "/home/praneeth/dissertation/GRF_MARL"

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.join(GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"))

from minimal_state import MinimalState
from enhanced_LightActionMask_5 import FeatureEncoder

import gfootball.env as football_env

env = football_env.create_environment(
    env_name='5_vs_5', representation='raw',
    number_of_left_players_agent_controls=4, render=False
)
raw_obs = env.reset()

encoder = FeatureEncoder()
state = MinimalState(n_player=5)
state.set_obs(raw_obs[0])

feat = encoder.encode_each(state)
print("Encoded feature vector length:", len(feat))
print("Expected (from desc.pkl):", 192)
assert len(feat) == 192, "MISMATCH -- do not proceed until this matches"
print("PASS: encoding matches expected dimension")

env.close()

import sys, os
GRF_MARL_ROOT = "/home/urstr/dissertation/GRF_MARL"
PROJECT_ROOT = "/home/urstr/dissertation/context-aware-magail-grf"
sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))

import gfootball.env as football_env

env = football_env.create_environment(
env_name="5_vs_5_d06", representation="raw",
number_of_left_players_agent_controls=4,
number_of_right_players_agent_controls=0,
render=False,
)
raw_obs = env.reset()
for i in range(4):
    print(f"raw_obs[{i}]['active'] = {raw_obs[i]['active']} (expect {i+1})")
env.close()

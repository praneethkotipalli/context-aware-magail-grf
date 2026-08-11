import sys
sys.path.insert(0, "/home/praneeth/dissertation/football")
from gfootball.env import config
from gfootball.env import football_env

cfg = config.Config({
    'action_set': 'default',
    'players': ['keyboard:left_players=1'],
    'level': '5_vs_5_d06',
    'representation': 'raw',
    'real_time': True,
    'rewards': 'scoring',
})
env = football_env.FootballEnv(cfg)
env.render()
obs = env.reset()

print("Type of obs:", type(obs))
print("Length (if applicable):", len(obs) if hasattr(obs, '__len__') else "N/A")
if isinstance(obs, list):
    print("Type of obs[0]:", type(obs[0]))
    if isinstance(obs[0], dict):
        print("Keys:", list(obs[0].keys())[:10])
elif isinstance(obs, dict):
    print("Keys:", list(obs.keys())[:10])

env.close()

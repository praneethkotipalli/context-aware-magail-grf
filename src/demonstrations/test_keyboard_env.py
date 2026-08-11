import sys
sys.path.insert(0, "/home/praneeth/dissertation/football")

from gfootball.env import config
from gfootball.env import football_env

cfg_values = {
    'action_set': 'default',
    'players': ['keyboard:left_players=1'],
    'level': '5_vs_5',
    'representation': 'raw',
    'real_time': True,
    'rewards': 'scoring',
}

cfg = config.Config(cfg_values)
env = football_env.FootballEnv(cfg)
env.render()
obs = env.reset()

print("Env created OK, keyboard player active. Play for a few seconds...")
done = False
steps = 0
while not done and steps < 200:
    obs, reward, done, info = env.step([])
    steps += 1

print(f"Stepped {steps} times OK -- check whether a game window appeared and responded to WASD/arrows/E")
env.close()

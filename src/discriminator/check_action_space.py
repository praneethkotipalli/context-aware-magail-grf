"""check_action_space.py -- confirm env.action_space shape before writing
the random-policy recorder. Cheap, no actor/encoder needed."""
import sys, os
GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")
sys.path.insert(0, GRF_MARL_ROOT)

import gfootball.env as football_env

env = football_env.create_environment(
    env_name="5_vs_5_d06", representation="raw",
    number_of_left_players_agent_controls=4,
    number_of_right_players_agent_controls=0,
    render=False,
)

print("action_space:", env.action_space)
print("type:", type(env.action_space))

sample = env.action_space.sample()
print("sample():", sample)
print("sample type:", type(sample))

raw_obs = env.reset()
print(f"\nnumber of agent obs dicts from reset(): {len(raw_obs)}")

# Try stepping with whatever sample() gave us, see if it's directly usable
try:
    if hasattr(sample, '__len__') and len(sample) == 4:
        actions = list(sample)
    else:
        actions = [int(sample)] * 4  # fallback guess, only if sample() is scalar
    raw_obs2, reward, done, info = env.step(actions)
    print(f"\nstep() succeeded with actions={actions}")
    print(f"reward: {reward}, done: {done}")
except Exception as e:
    print(f"\nstep() FAILED with actions from sample(): {e}")

env.close()
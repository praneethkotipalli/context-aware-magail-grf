"""
inspect_observation.py
Inspects the GRF 5v5 simple115v2 observation structure to verify index alignment.
"""

import numpy as np
import gfootball.env as football_env

def inspect_grf_observation():
    print("=" * 60)
    print("GRF OBSERVATION INSPECTOR — 5v5 SCENARIO")
    print("=" * 60)
    
    env = football_env.create_environment(
        env_name='5_vs_5',
        representation='simple115v2',
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=False
    )
    
    obs = env.reset()
    
    print(f"\n[INFO] Obs Type:   {type(obs)}")
    print(f"[INFO] Obs Shape:  {obs.shape}")
    print(f"[INFO] Action Space: {env.action_space}")
    
    agent_0_obs = obs[0]
    print(f"\n[INFO] Controllable Agent 0 Vector Length: {len(agent_0_obs)}")
    
    print("\n--- FIRST 30 VALUES (Player Coordinates Region) ---")
    for i in range(30):
        print(f"Index [{i:3d}]: {agent_0_obs[i]:.4f}")
        
    print("\n--- INDICES 85 to 95 (Ball & Game State Region) ---")
    for i in range(85, 96):
        print(f"Index [{i:3d}]: {agent_0_obs[i]:.4f}")

    print("\n[STEP] Taking 3 random steps to test environment iteration...")
    for step in range(3):
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        print(f"  Step {step + 1} | Actions taken: {actions} | Reward: {reward} | Done: {done}")

    env.close()
    print("\n[DONE] Inspection completed cleanly.")

if __name__ == '__main__':
    inspect_grf_observation()

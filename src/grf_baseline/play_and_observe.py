"""
play_and_observe.py
Launches Google Research Football in 5v5 mode with LIVE 3D RENDERING.
"""

import time
import numpy as np
import gfootball.env as football_env

def run_match():
    print("==========================================================")
    print("       LAUNCHING GRF 5v5 LIVE VISUAL SIMULATION           ")
    print("==========================================================")
    
    # render=True enables OpenGL 3D pitch rendering
    env = football_env.create_environment(
        env_name='5_vs_5',
        representation='simple115v2',
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=True
    )
    
    obs = env.reset()
    print(f"\n[INIT] Visual Match Loaded!")
    print(f"[INIT] Observation Shape: {obs.shape}")
    
    steps = 200
    print(f"\n[SIMULATION] Rendering {steps} match steps...\n")
    
    for step in range(1, steps + 1):
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        
        agent_0_pos = obs[0][:2]
        ball_pos = obs[0][88:90]
        possession = (obs[0][94] == 1.0)
        
        print(f"Step {step:03d} | Agent 0 (x,y): {agent_0_pos} | Ball (x,y): {ball_pos} | Has Ball: {possession}")
        
        # 0.04s sleep yields ~25 FPS live match speed
        time.sleep(0.04)
        
        if done:
            print("\n[EVENT] Goal/Episode End! Resetting...")
            obs = env.reset()

    env.close()
    print("\n==========================================================")
    print("              SIMULATION COMPLETED CLEANLY                ")
    print("==========================================================")

if __name__ == '__main__':
    run_match()

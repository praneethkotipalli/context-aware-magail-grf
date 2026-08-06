"""
src/grf_baseline/compare_checkpoints.py

Memory-safe, correctly-parallelized checkpoint comparison with live progress
reporting. Uses a bounded worker pool (not one process per checkpoint) to
avoid OOM, pins each worker to a single CPU thread to avoid PyTorch thread
oversubscription, and reports progress via a shared queue so you can see
episodes completing in real time instead of waiting silently until the end.

Usage: python src/grf_baseline/compare_checkpoints.py --episodes 10 --workers 3
"""

import os
import sys
import argparse
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

GRF_MARL_ROOT = "/home/praneeth/dissertation/GRF_MARL"
PROJECT_ROOT = "/home/praneeth/dissertation/context-aware-magail-grf"

CANDIDATES = [
    "GKBug_v0", "GKBug_v1", "GKBug_v2", "GKBug_v3",
    "PassingMain_v1", "PassingMain_v2",
]


def run_checkpoint_worker(policy_name, n_episodes, progress_queue):
    """
    Runs in its own process. Pinned to 1 CPU thread to prevent PyTorch
    from oversubscribing cores when multiple workers run concurrently.
    Reports per-episode progress via progress_queue so the main process
    can print live updates instead of staying silent until completion.
    """
    import torch
    torch.set_num_threads(1)

    sys.path.insert(0, GRF_MARL_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
    sys.path.insert(0, os.path.join(
        GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"
    ))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "metrics"))

    import gfootball.env as football_env
    from minimal_state import MinimalState
    from enhanced_LightActionMask_5 import FeatureEncoder
    from calculator import compute_sap, compute_sbf, compute_mecha, compute_cv

    actor_path = os.path.join(
        GRF_MARL_ROOT, "light_malib", "trained_models", "gr_football",
        "5_vs_5", policy_name, "actor.pt"
    )

    try:
        actor = torch.load(actor_path, map_location="cpu")
        actor.eval()
    except Exception as e:
        progress_queue.put(("ERROR", policy_name, str(e)))
        return {"policy": policy_name, "error": str(e)}

    encoder = FeatureEncoder()
    env = football_env.create_environment(
        env_name="5_vs_5", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0,
        render=False,
    )

    episode_results = []
    t_start = time.time()

    for ep in range(n_episodes):
        raw_obs = env.reset()
        sticky_sprint_log, position_log, possession_log = [], [], []
        step = 0
        done = False
        max_steps = 3000

        while not done and step < max_steps:
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

            sticky_sprint_log.append([int(a["sticky_actions"][8]) for a in raw_obs])
            position_log.append(np.array(raw_obs[0]["left_team"])[:5])
            possession_log.append(bool(raw_obs[0]["ball_owned_team"] == 0))

            raw_obs, reward, done, info = env.step(actions)
            step += 1

        final_score = raw_obs[0]["score"]
        left_score, right_score = int(final_score[0]), int(final_score[1])

        sticky_arr = np.array(sticky_sprint_log)
        pos_arr = np.array(position_log)
        poss_arr = np.array(possession_log)

        ep_result = {
            "win": int(left_score > right_score),
            "draw": int(left_score == right_score),
            "loss": int(left_score < right_score),
            "sap": compute_sap(sticky_arr),
            "sbf": compute_sbf(sticky_arr),
            "mecha": compute_mecha(pos_arr, poss_arr),
            "cv": compute_cv(pos_arr),
        }
        episode_results.append(ep_result)

        # Live progress -- this is the fix for "no visibility until the end"
        progress_queue.put((
            "PROGRESS", policy_name,
            f"episode {ep+1}/{n_episodes} | sap={ep_result['sap']:.1f}% | win={ep_result['win']}"
        ))

    env.close()
    elapsed = time.time() - t_start

    df = pd.DataFrame(episode_results)
    result = {
        "policy": policy_name,
        "win_rate": df["win"].mean(),
        "draw_rate": df["draw"].mean(),
        "sap_mean": df["sap"].mean(),
        "sap_std": df["sap"].std(),
        "sbf_mean": df["sbf"].mean(),
        "mecha_mean": df["mecha"].mean(),
        "cv_mean": df["cv"].mean(),
        "n_episodes": n_episodes,
        "elapsed_sec": elapsed,
    }
    progress_queue.put(("DONE", policy_name, result))
    return result


def worker_entrypoint(args_tuple):
    policy_name, n_episodes, progress_queue = args_tuple
    return run_checkpoint_worker(policy_name, n_episodes, progress_queue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3,
                         help="Max concurrent processes. Keep this <= physical cores "
                              "and low enough to avoid OOM on memory-constrained machines.")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    progress_queue = manager.Queue()

    print(f"Comparing {len(CANDIDATES)} checkpoints, {args.episodes} episodes each, "
          f"max {args.workers} concurrent workers (bounded pool -- memory-safe)")
    print(f"Candidates: {CANDIDATES}")
    print("-" * 90)

    t0 = time.time()
    task_args = [(policy_name, args.episodes, progress_queue) for policy_name in CANDIDATES]

    with mp.Pool(processes=args.workers) as pool:
        async_result = pool.map_async(worker_entrypoint, task_args)

        # Live progress printing loop -- drains the queue while workers run
        completed_policies = set()
        while not async_result.ready() or not progress_queue.empty():
            try:
                msg_type, policy_name, payload = progress_queue.get(timeout=1.0)
                if msg_type == "PROGRESS":
                    print(f"[{policy_name}] {payload}", flush=True)
                elif msg_type == "DONE":
                    print(f"[{policy_name}] *** COMPLETE *** "
                          f"win_rate={payload['win_rate']:.2f} "
                          f"sap={payload['sap_mean']:.1f}% "
                          f"({payload['elapsed_sec']:.1f}s)", flush=True)
                    completed_policies.add(policy_name)
                elif msg_type == "ERROR":
                    print(f"[{policy_name}] *** ERROR *** {payload}", flush=True)
            except Exception:
                continue  # queue empty this tick, loop again

        results = async_result.get()

    total_elapsed = time.time() - t0
    print("-" * 90)
    print(f"All {len(CANDIDATES)} checkpoints complete in {total_elapsed:.1f}s wall-clock")

    df = pd.DataFrame(results)
    df = df.sort_values("win_rate", ascending=False)

    output_dir = os.path.join(PROJECT_ROOT, "results", "baseline")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "checkpoint_comparison.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("CHECKPOINT COMPARISON -- sorted by win_rate")
    print("=" * 90)
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
"""
src/grf_baseline/villain_numbers_blackwell.py

Full villain-number characterization (500 episodes per policy) across the
4 selected competent 5v5 checkpoints. Task-chunked parallelism across a
bounded worker pool, crash-safe per-chunk CSV logging, live progress via
queue, W&B logging from the main process only.
"""

import os
import sys
import argparse
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import wandb

PROJECT_ROOT = "/home/u5749464/dissertation/context-aware-magail-grf"
GRF_MARL_ROOT = "/home/u5749464/dissertation/GRF_MARL"

POLICIES = ["GKBug_v0", "GKBug_v2", "GKBug_v3", "PassingMain_v2"]


def run_chunk(policy_name, chunk_id, n_episodes, progress_queue):
    import torch
    torch.set_num_threads(1)

    sys.path.insert(0, GRF_MARL_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
    sys.path.insert(0, os.path.join(
        GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"
    ))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "metrics"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "utils"))

    import gfootball.env as football_env
    from minimal_state import MinimalState
    from enhanced_LightActionMask_5 import FeatureEncoder
    from calculator import compute_sap, compute_sbf, compute_mecha, compute_cv
    from logging_utils import IncrementalCSVLogger

    actor_path = os.path.join(
        GRF_MARL_ROOT, "light_malib", "trained_models", "gr_football",
        "5_vs_5", policy_name, "actor.pt"
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

    chunk_csv_path = os.path.join(
        PROJECT_ROOT, "results", "baseline", "villain",
        f"{policy_name}_chunk{chunk_id}.csv"
    )
    fieldnames = ["policy", "chunk_id", "episode", "win", "draw", "loss",
                  "sap", "sbf", "mecha", "cv"]
    logger = IncrementalCSVLogger(chunk_csv_path, fieldnames)

    chunk_results = []

    for ep in range(n_episodes):
        raw_obs = env.reset()
        sticky_sprint_log, position_log, possession_log = [], [], []
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

        row = {
            "policy": policy_name, "chunk_id": chunk_id, "episode": ep,
            "win": int(left_score > right_score),
            "draw": int(left_score == right_score),
            "loss": int(left_score < right_score),
            "sap": compute_sap(sticky_arr),
            "sbf": compute_sbf(sticky_arr),
            "mecha": compute_mecha(pos_arr, poss_arr),
            "cv": compute_cv(pos_arr),
        }
        logger.log(row)
        chunk_results.append(row)

        progress_queue.put(("PROGRESS", policy_name, chunk_id,
                             f"{ep+1}/{n_episodes} sap={row['sap']:.1f}% win={row['win']}"))

    logger.close()
    env.close()

    progress_queue.put(("CHUNK_DONE", policy_name, chunk_id, len(chunk_results)))
    return chunk_results


def worker_entrypoint(task):
    policy_name, chunk_id, n_episodes, progress_queue = task
    return run_chunk(policy_name, chunk_id, n_episodes, progress_queue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-policy", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--wandb-project", type=str, default="dissertation-grf-alignment")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    run = wandb.init(
        project=args.wandb_project,
        name="villain-numbers-4policy-500ep_d06",
        config={
            "policies": POLICIES,
            "episodes_per_policy": args.episodes_per_policy,
            "chunk_size": args.chunk_size,
            "workers": args.workers,
        },
    )

    tasks_meta = []
    for policy_name in POLICIES:
        n_chunks = (args.episodes_per_policy + args.chunk_size - 1) // args.chunk_size
        for chunk_id in range(n_chunks):
            remaining = args.episodes_per_policy - chunk_id * args.chunk_size
            this_chunk_size = min(args.chunk_size, remaining)
            tasks_meta.append((policy_name, chunk_id, this_chunk_size))

    print(f"Total tasks: {len(tasks_meta)} chunks across {len(POLICIES)} policies, "
          f"{args.workers} concurrent workers")
    print(f"({args.episodes_per_policy} episodes/policy, {args.chunk_size} episodes/chunk)")
    print("-" * 90)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    tasks = [(p, c, n, progress_queue) for p, c, n in tasks_meta]

    t0 = time.time()
    chunk_done_count = {p: 0 for p in POLICIES}
    total_chunks = {p: sum(1 for pp, _, _ in tasks_meta if pp == p) for p in POLICIES}

    with mp.Pool(processes=args.workers) as pool:
        async_result = pool.map_async(worker_entrypoint, tasks)

        while not async_result.ready() or not progress_queue.empty():
            try:
                msg = progress_queue.get(timeout=1.0)
                if msg[0] == "PROGRESS":
                    _, policy_name, chunk_id, detail = msg
                    print(f"[{policy_name} chunk{chunk_id}] {detail}", flush=True)
                elif msg[0] == "CHUNK_DONE":
                    _, policy_name, chunk_id, n = msg
                    chunk_done_count[policy_name] += 1
                    pct = 100 * chunk_done_count[policy_name] / total_chunks[policy_name]
                    print(f"[{policy_name}] chunk {chunk_id} complete "
                          f"({chunk_done_count[policy_name]}/{total_chunks[policy_name]} "
                          f"chunks, {pct:.0f}%)", flush=True)
                    wandb.log({f"progress/{policy_name}_pct": pct})
            except Exception:
                continue

        all_chunk_results = async_result.get()

    total_elapsed = time.time() - t0
    print("-" * 90)
    print(f"All chunks complete in {total_elapsed/60:.1f} minutes")

    all_rows = [row for chunk in all_chunk_results for row in chunk]
    df = pd.DataFrame(all_rows)

    merged_path = os.path.join(PROJECT_ROOT, "results", "baseline", "villain_numbers_500ep_d06.csv")
    df.to_csv(merged_path, index=False)

    summary = df.groupby("policy").agg(
        win_rate=("win", "mean"),
        draw_rate=("draw", "mean"),
        sap_mean=("sap", "mean"),
        sap_std=("sap", "std"),
        sbf_mean=("sbf", "mean"),
        mecha_mean=("mecha", "mean"),
        cv_mean=("cv", "mean"),
        n_episodes=("episode", "count"),
    ).sort_values("win_rate", ascending=False)

    summary_path = os.path.join(PROJECT_ROOT, "results", "baseline", "villain_numbers_summary_d06.csv")
    summary.to_csv(summary_path)

    print("\n" + "=" * 90)
    print("FINAL VILLAIN NUMBERS -- 500 episodes per policy")
    print("=" * 90)
    print(summary.to_string())
    print(f"\nFull data: {merged_path}")
    print(f"Summary:   {summary_path}")

    for policy_name in POLICIES:
        row = summary.loc[policy_name]
        wandb.log({
            f"final/{policy_name}_win_rate": row["win_rate"],
            f"final/{policy_name}_sap_mean": row["sap_mean"],
            f"final/{policy_name}_mecha_mean": row["mecha_mean"],
        })

    wandb.finish()


if __name__ == "__main__":
    main()

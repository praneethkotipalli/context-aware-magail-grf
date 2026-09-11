"""
record_condition_rollouts.py

60 episodes x 4 representative seeds x 4 conditions = 960 episodes,
concurrent (one process per condition,seed = 16 workers). Matches the
expert .npz schema exactly (record_session.py / verifyingepisodes.ipynb),
with ONE documented difference: action_captured is (T,4) here (all 4
policy-controlled players), not (T,) like the single-human expert
recordings.

CONTEXT_REGION IS COMPUTED CORRECTLY HERE -- record_session.py's own
classify_region() has the t_norm direction backwards (flagged in chat).
This script uses the SAME canonical convention the training pipeline and
the analysis notebook both use: elapsed_min = (1-t_norm)*90, late means
elapsed_min >= 70 (equivalently t_norm <= 0.2222).

REPRESENTATIVE SEED SELECTION: reads all_runs_raw.json (the flat,
per-seed 500-episode final-eval results) and picks the 4 seeds per
condition whose sap_mean is closest to that condition's own mean --
"typical," not outlier, seeds. Falls back to seeds [0,1,2,3] with a
loud warning if that file isn't available yet.

Run:
    python record_condition_rollouts.py \
        --launch-id matrix_0911ecd0 --episodes 60 --n-seeds 4 --workers 16
"""
import argparse, csv, datetime, json, os, subprocess, sys, time
import numpy as np
import torch

CONDITIONS = ["NC", "C", "SC", "C+KL"]


def classify_region(t_norm):
    """CORRECT direction -- matches classify_bin / the analysis notebook's
    classify_time_zone, NOT record_session.py's own (inverted) version."""
    elapsed_min = (1.0 - t_norm) * 90.0
    if elapsed_min < 20:
        return "early"
    if elapsed_min < 70:
        return "mid"
    return "late"


def pick_representative_seeds(launch_id, condition, n_seeds, result_dir):
    """Picks n_seeds seeds whose 500ep final-eval sap_mean is closest to
    the condition's own mean. Falls back to [0..n_seeds-1] if the final
    evaluation data isn't available."""
    raw_path = os.path.join(result_dir, "all_runs_raw.json")
    if not os.path.exists(raw_path):
        print(f"  WARNING: {raw_path} not found -- falling back to seeds "
              f"{list(range(n_seeds))} for condition {condition} (not "
              f"representative-selected)")
        return list(range(n_seeds))

    with open(raw_path) as f:
        all_runs = json.load(f)
    rows = [r for r in all_runs if r.get("condition_label") == condition
           and r.get("status") != "missing" and r.get("sap_mean") is not None]
    if not rows:
        print(f"  WARNING: no valid final-eval rows for {condition} -- "
              f"falling back to seeds {list(range(n_seeds))}")
        return list(range(n_seeds))

    saps = [r["sap_mean"] for r in rows]
    mean_sap = np.mean(saps)
    ranked = sorted(rows, key=lambda r: abs(r["sap_mean"] - mean_sap))
    picked = [r["seed"] for r in ranked[:n_seeds]]
    print(f"  {condition}: picked seeds {picked} "
          f"(condition mean SAP={mean_sap:.2f}%, "
          f"picked SAPs={[round(r['sap_mean'],2) for r in ranked[:n_seeds]]})")
    return picked


def record_worker_main():
    """Runs INSIDE a subprocess -- one (condition, seed) job, n_episodes
    recorded sequentially within this one process (checkpoint/env load
    cost paid once, not per episode)."""
    import argparse as ap_mod
    p = ap_mod.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start-episode-id", type=int, default=0)
    args = p.parse_args()

    from finetune_loop import build_env, ACTOR_PATH, MAX_STEPS
    from context_conditioned_policy import ContextConditionedActor
    from minimal_state import MinimalState
    from enhanced_LightActionMask_5 import FeatureEncoder

    frozen_actor_structure = torch.load(ACTOR_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor_structure)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    encoder = FeatureEncoder()
    env = build_env()
    os.makedirs(args.out_dir, exist_ok=True)

    coverage = {"early": 0, "mid": 0, "late": 0}
    score_coverage = {(r, z): 0 for r in ("early","mid","late") for z in ("win","draw","loss")}
    summary_rows = []

    for ep_i in range(args.episodes):
        ep_id = args.start_episode_id + ep_i
        raw_obs = env.reset()
        steps_left_l, score_l, score_r = [], [], []
        left_team_l, left_dir_l, right_team_l, right_dir_l = [], [], [], []
        ball_l, ball_dir_l, owned_team_l, owned_player_l = [], [], [], []
        sticky_l, action_l, poss_l, region_l = [], [], [], []

        done, step = False, 0
        while not done and step < MAX_STEPS:
            o = raw_obs[0]
            t_norm = o['steps_left'] / 3001.0
            dscore = int(o['score'][0]) - int(o['score'][1])
            region = classify_region(t_norm)
            score_zone = 'win' if dscore > 0 else ('loss' if dscore < 0 else 'draw')
            coverage[region] += 1
            score_coverage[(region, score_zone)] += 1

            steps_left_l.append(o['steps_left'])
            score_l.append(o['score'][0]); score_r.append(o['score'][1])
            left_team_l.append(np.array(o['left_team'], dtype=np.float64))
            left_dir_l.append(np.array(o['left_team_direction'], dtype=np.float64))
            right_team_l.append(np.array(o['right_team'], dtype=np.float64))
            right_dir_l.append(np.array(o['right_team_direction'], dtype=np.float64))
            ball_l.append(np.array(o['ball'], dtype=np.float64))
            ball_dir_l.append(np.array(o['ball_direction'], dtype=np.float64))
            owned_team_l.append(o['ball_owned_team']); owned_player_l.append(o['ball_owned_player'])
            sticky_l.append(np.array(raw_obs[0]['sticky_actions'], dtype=np.uint8))
            poss_l.append(bool(o['ball_owned_team'] == 0))
            region_l.append(region)

            ctx_t = torch.as_tensor([t_norm, np.clip(dscore, -3, 3)/3.0],
                                    dtype=torch.float32).unsqueeze(0).repeat(4, 1)
            obs_192, masks = [], []
            for i in range(4):
                s = MinimalState(n_player=5); s.set_obs(raw_obs[i])
                obs_192.append(encoder.encode_each(s))
                masks.append(encoder.get_available_actions(raw_obs[i], 0.0, []))
            obs_t = torch.as_tensor(np.stack(obs_192), dtype=torch.float32)
            mask_t = torch.as_tensor(np.stack(masks), dtype=torch.float32)

            with torch.no_grad():
                actions, _, _, _ = actor(obs_t, ctx_t, mask_t, explore=False)
            action_l.append(actions.numpy().astype(np.int64))  # (4,) all controlled players

            raw_obs, _, done, _ = env.step(actions.tolist())
            step += 1

        n_steps = step
        npz_path = os.path.join(args.out_dir, f"{args.condition}_seed{args.seed}_ep{ep_id}.npz")
        np.savez_compressed(
            npz_path,
            episode_id=ep_id, context_label=args.condition, seed=args.seed,
            steps_left=np.array(steps_left_l), score_left=np.array(score_l), score_right=np.array(score_r),
            left_team=np.array(left_team_l), left_team_direction=np.array(left_dir_l),
            right_team=np.array(right_team_l), right_team_direction=np.array(right_dir_l),
            ball=np.array(ball_l), ball_direction=np.array(ball_dir_l),
            ball_owned_team=np.array(owned_team_l), ball_owned_player=np.array(owned_player_l),
            sticky_actions=np.array(sticky_l), action_captured=np.array(action_l),  # (T,4)
            has_possession=np.array(poss_l), context_region=np.array(region_l, dtype='<U16'),
        )
        summary_rows.append({"episode_id": ep_id, "condition": args.condition, "seed": args.seed,
                            "n_steps": n_steps, "npz_path": npz_path})
        print(f"  [{args.condition} seed{args.seed}] ep{ep_id} steps={n_steps}")

    env.close()
    print(f"[{args.condition} seed{args.seed}] DONE. region coverage: {coverage}")
    print(f"[{args.condition} seed{args.seed}] late-win={score_coverage[('late','win')]} "
          f"late-loss={score_coverage[('late','loss')]} late-draw={score_coverage[('late','draw')]}")
    sc_serializable = {f"{r}_{z}": v for (r, z), v in score_coverage.items()}
    with open(os.path.join(args.out_dir, f"_summary_{args.condition}_seed{args.seed}.json"), "w") as f:
        json.dump({"coverage": coverage, "score_coverage": sc_serializable,
                   "episodes": summary_rows}, f, indent=2)


def orchestrate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-id", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    result_dir = os.path.join("results_v2", "stage3_full_matrix", args.launch_id)
    out_root = os.path.join("results_v2", "stage4_rollouts", args.launch_id)
    episodes_dir = os.path.join(out_root, "episodes")
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("picking representative seeds per condition...")
    seed_map = {c: pick_representative_seeds(args.launch_id, c, args.n_seeds, result_dir)
               for c in CONDITIONS}
    manifest = {"launch_id": args.launch_id, "episodes_per_seed": args.episodes,
               "seed_selection": seed_map,
               "utc": datetime.datetime.utcnow().isoformat() + "Z"}
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    jobs = []
    for cond in CONDITIONS:
        for seed in seed_map[cond]:
            ckpt_path = f"ablation_final-{cond}-{args.launch_id}_seed{seed}.pt"
            jobs.append((cond, seed, ckpt_path))

    print(f"\n{len(jobs)} jobs ({args.workers}-way concurrency), "
          f"{args.episodes} episodes each\n")

    queue, running, completed = list(jobs), [], []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < args.workers:
            cond, seed, ckpt_path = queue.pop(0)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {cond} seed{seed}: checkpoint {ckpt_path} not found")
                continue
            log_path = os.path.join(log_dir, f"{cond}_seed{seed}.log")
            logf = open(log_path, "w")
            cmd = [sys.executable, __file__, "--worker",
                   "--checkpoint", ckpt_path, "--condition", cond, "--seed", str(seed),
                   "--episodes", str(args.episodes), "--out-dir", episodes_dir]
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            print(f"  launched {cond} seed{seed}  [{len(running)+1}/{args.workers} slots]")
            running.append({"cond": cond, "seed": seed, "proc": proc, "logf": logf, "start": time.time()})
        time.sleep(5)
        still = []
        for j in running:
            ret = j["proc"].poll()
            if ret is None:
                still.append(j)
            else:
                j["logf"].close()
                el = time.time() - j["start"]
                print(f"  finished {j['cond']} seed{j['seed']}  ({el/60:.1f}min, "
                      f"{'OK' if ret == 0 else f'EXIT {ret}'})")
                completed.append({**j, "returncode": ret})
        running = still

    total = time.time() - t0
    print(f"\nALL DONE in {total/60:.1f}min -- "
          f"{sum(1 for c in completed if c['returncode']==0)}/{len(jobs)} jobs succeeded")

    # aggregate coverage across all jobs
    total_coverage = {"early": 0, "mid": 0, "late": 0}
    total_score_coverage = {"late_win": 0, "late_draw": 0, "late_loss": 0}
    for cond in CONDITIONS:
        for seed in seed_map[cond]:
            sp = os.path.join(episodes_dir, f"_summary_{cond}_seed{seed}.json")
            if os.path.exists(sp):
                with open(sp) as f:
                    d = json.load(f)
                for k in total_coverage:
                    total_coverage[k] += d["coverage"].get(k, 0)
                for k in total_score_coverage:
                    total_score_coverage[k] += d.get("score_coverage", {}).get(k, 0)
    print(f"\nTOTAL region coverage across all recorded episodes: {total_coverage}")
    print(f"TOTAL late-win={total_score_coverage['late_win']} "
          f"late-loss={total_score_coverage['late_loss']} "
          f"late-draw={total_score_coverage['late_draw']}")
    print(f"(compare late-loss/late-win directly against expert corpus's "
          f"~16,650 late-loss + ~10,000 late-win steps -- this is now a fair, "
          f"like-for-like comparison, not the time-only tally from before)")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        record_worker_main()
    else:
        orchestrate()
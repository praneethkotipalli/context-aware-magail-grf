"""
run_final_evaluation.py

Runs the 500-episode final evaluation on all 32 already-trained
checkpoints (no re-training -- checkpoints from the completed training
phase are used as-is), then produces THREE DISTINCT file types:

  1. PER-RUN:      final_eval_<run_name>.json      (32 files, one per run)
                   Written directly by evaluate_finetuned_checkpoint.py.
                   Raw 500-episode result for exactly one (condition, seed).

  2. AGGREGATION:  all_runs_raw.json                (1 file)
                   A FLAT LIST of all 32 per-run results, each tagged with
                   condition_label + seed. This is the RAW DATA TABLE --
                   load it straight into pandas for significance testing
                   (groupby condition_label, compare distributions). No
                   summary statistics here, only raw per-seed values --
                   collapsing to means here would make later stats tests
                   impossible.

  3. SUMMARY:      condition_summary.json           (1 file)
                   Per-condition descriptive stats ONLY (mean/sd/se/n) for
                   each metric. For reporting and plotting (bar charts,
                   radar charts) -- NOT for significance testing, since it
                   has no per-seed values. Use all_runs_raw.json for that.

Also logs a W&B summary table and does a best-effort git push at the end.

Run:
    python run_final_evaluation.py --launch-id matrix_0911ecd0 --workers 32
"""
import argparse, datetime, json, os, subprocess, sys, time
import statistics as st
import wandb

CONDITIONS = ["NC", "C", "SC", "C+KL"]
SEEDS = list(range(8))
FINAL_EVAL_EPISODES = 500
MANIFEST_PATH = os.path.join("results_v2", "MANIFEST.json")
METRICS = ["win_rate", "sap_mean", "mecha_mean", "csi_sap_proxy"]


def condition_for(cond_name, launch_id):
    return f"final-{cond_name}-{launch_id}"


def run_name_for(cond_name, seed, launch_id):
    return f"{condition_for(cond_name, launch_id)}_seed{seed}"


def build_plan():
    plan = [(c, s) for c in CONDITIONS for s in SEEDS]
    plan.sort(key=lambda x: (x[1], CONDITIONS.index(x[0])))
    return plan


def launch_eval_job(cond_name, seed, episodes, launch_id, result_dir, log_dir):
    run_name = run_name_for(cond_name, seed, launch_id)
    ckpt_path = f"ablation_{run_name}.pt"
    out_path = os.path.join(result_dir, f"final_eval_{run_name}.json")
    log_path = os.path.join(log_dir, f"finaleval_{run_name}.log")
    logf = open(log_path, "w")

    if not os.path.exists(ckpt_path):
        logf.write(f"MISSING CHECKPOINT: {ckpt_path} not found in {os.getcwd()}\n")
        logf.close()
        class _Dead:
            def poll(self): return 1
        return {"cond": cond_name, "seed": seed, "run_name": run_name, "out_path": out_path,
                "proc": _Dead(), "logf": None, "start": time.time()}

    cmd = [sys.executable, "evaluate_finetuned_checkpoint.py",
           "--checkpoint", ckpt_path, "--episodes", str(episodes),
           "--out", out_path, "--group", cond_name]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"cond": cond_name, "seed": seed, "run_name": run_name, "out_path": out_path,
            "proc": proc, "logf": logf, "start": time.time()}


def run_pool(jobs_spec, n_workers, launch_id, result_dir, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    queue = list(jobs_spec)
    running, completed = [], []
    print(f"\n{len(queue)} jobs, {n_workers}-way concurrency, "
          f"{FINAL_EVAL_EPISODES} episodes each")

    while queue or running:
        while queue and len(running) < n_workers:
            cond, seed, episodes = queue.pop(0)
            job = launch_eval_job(cond, seed, episodes, launch_id, result_dir, log_dir)
            print(f"  launched {job['run_name']}  [{len(running)+1}/{n_workers} slots]")
            running.append(job)
        time.sleep(5)
        still = []
        for job in running:
            ret = job["proc"].poll()
            if ret is None:
                still.append(job)
            else:
                if job["logf"]:
                    job["logf"].close()
                elapsed = time.time() - job["start"]
                print(f"  finished {job['run_name']}  ({elapsed/60:.1f}min, "
                      f"{'OK' if ret == 0 else f'EXIT {ret}'})")
                job["returncode"] = ret
                completed.append(job)
        running = still
    return completed


def build_aggregation_and_summary(plan, launch_id, result_dir):
    """Builds files 2 and 3. File 1 (per-run) already exists on disk from
    evaluate_finetuned_checkpoint.py -- this function only READS those,
    never rewrites them, so raw per-run data always stays exactly what
    the eval subprocess itself produced."""

    # ---- AGGREGATION: flat list, raw values, no collapsing ----
    all_runs = []
    for cond, seed in plan:
        run_name = run_name_for(cond, seed, launch_id)
        path = os.path.join(result_dir, f"final_eval_{run_name}.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            r.setdefault("condition_label", cond)
            r.setdefault("seed", seed)
            all_runs.append(r)
        else:
            print(f"  MISSING per-run result: {path}")
            all_runs.append({"condition_label": cond, "seed": seed, "run_name": run_name,
                            "status": "missing"})

    agg_path = os.path.join(result_dir, "all_runs_raw.json")
    with open(agg_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    n_ok = sum(1 for r in all_runs if r.get("status") != "missing")
    print(f"\nwrote {agg_path}  ({n_ok}/{len(plan)} runs present)")

    # ---- SUMMARY: descriptive stats per condition, metric-by-metric ----
    summary = []
    for cond in CONDITIONS:
        rows = [r for r in all_runs if r["condition_label"] == cond and r.get("status") != "missing"]
        entry = {"condition": cond, "n_seeds": len(rows),
                "seeds": [r["seed"] for r in rows]}
        for metric in METRICS:
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            if vals:
                mean = st.mean(vals)
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                se = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
                entry[metric] = {"mean": mean, "sd": sd, "se": se, "n": len(vals)}
                # NOTE: no "values" here deliberately -- this file is for
                # reporting/plotting only. Raw values live in
                # all_runs_raw.json, which is what stats tests should read.
            else:
                entry[metric] = None
        summary.append(entry)

    summary_path = os.path.join(result_dir, "condition_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")

    print(f"\n{'condition':>8s} {'n':>2s} {'win mean':>9s} {'win se':>7s} "
          f"{'SAP mean':>9s} {'SAP se':>7s}")
    for row in summary:
        w, s = row.get("win_rate"), row.get("sap_mean")
        if w and s:
            print(f"{row['condition']:>8s} {row['n_seeds']:>2d} {w['mean']:>9.3f} "
                  f"{w['se']:>7.3f} {s['mean']:>9.2f} {s['se']:>7.2f}")
        else:
            print(f"{row['condition']:>8s}  incomplete ({row['n_seeds']} seeds)")

    return all_runs, summary


def log_summary_to_wandb(summary, launch_id):
    wandb.init(project="magail-c-v2-finaleval", name=f"SUMMARY_{launch_id}",
              group="summary", reinit=True)
    cols = ["condition", "n_seeds", "win_mean", "win_se", "sap_mean", "sap_se",
            "csi_mean", "csi_se"]
    wb_table = wandb.Table(columns=cols)
    for row in summary:
        w, s, c = row.get("win_rate"), row.get("sap_mean"), row.get("csi_sap_proxy")
        wb_table.add_data(row["condition"], row["n_seeds"],
                          w["mean"] if w else None, w["se"] if w else None,
                          s["mean"] if s else None, s["se"] if s else None,
                          c["mean"] if c else None, c["se"] if c else None)
    wandb.log({"ablation_summary": wb_table})
    wandb.finish()
    print(f"logged summary table to W&B (project magail-c-v2-finaleval, SUMMARY_{launch_id})")


def git_push():
    print(f"\n{'='*70}\nGIT PUSH\n{'='*70}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        commit = subprocess.run(["git", "commit", "-m", "final evaluation: per-run + aggregation + summary"],
                                capture_output=True, text=True)
        print(f"  {commit.stdout}{commit.stderr}".strip())
        push = subprocess.run(["git", "push"], capture_output=True, text=True)
        print("  pushed successfully." if push.returncode == 0
              else f"  git push FAILED (not fatal): {push.stderr}")
    except Exception as e:
        print(f"  git push step raised {e!r} -- not fatal.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-id", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=FINAL_EVAL_EPISODES)
    args = ap.parse_args()

    result_dir = os.path.join("results_v2", "stage3_full_matrix", args.launch_id)
    log_dir = os.path.join(result_dir, "eval_logs")
    if not os.path.isdir(result_dir):
        print(f"FAIL: {result_dir} does not exist -- check --launch-id.")
        return
    os.makedirs(result_dir, exist_ok=True)

    plan = build_plan()
    jobs_spec = [(c, s, args.episodes) for (c, s) in plan]

    t0 = time.time()
    completed = run_pool(jobs_spec, args.workers, args.launch_id, result_dir, log_dir)
    print(f"\nphase done in {(time.time()-t0)/3600:.2f}h  |  "
          f"succeeded: {sum(1 for c in completed if c['returncode']==0)}/{len(plan)}")
    if all(c["returncode"] != 0 for c in completed):
        print(f"\nWARNING: every job failed. Check {log_dir}/ before trusting anything below.")

    all_runs, summary = build_aggregation_and_summary(plan, args.launch_id, result_dir)
    log_summary_to_wandb(summary, args.launch_id)

    os.makedirs("results_v2", exist_ok=True)
    manifest = json.load(open(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else []
    manifest.append({"launch_id": args.launch_id, "stage": "final_evaluation",
                     "utc": datetime.datetime.utcnow().isoformat() + "Z",
                     "n_runs": len(plan), "episodes": args.episodes})
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)

    git_push()
    print(f"\nDONE -- {result_dir}/")
    print(f"  per-run:     final_eval_*.json  (32 files)")
    print(f"  aggregation: all_runs_raw.json  (flat list, raw values -- use for stats tests)")
    print(f"  summary:     condition_summary.json  (descriptive stats -- use for plots)")


if __name__ == "__main__":
    main()
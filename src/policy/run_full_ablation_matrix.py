"""
run_full_ablation_matrix.py

4 conditions x 8 seeds = 32 runs, 32-way concurrency, 300 iterations each,
then a SEPARATE 500-episode final evaluation per run (matching the MAPPO
baseline's own 500-episode standard for direct comparability) using
evaluate_finetuned_checkpoint.py. Individual JSON per run, cross-seed
aggregation table per condition, auto git push at the very end.

CONDITIONS:
    NC    disc_int_context=zero    use_kl=0
    C     disc_int_context=true    use_kl=0
    SC    disc_int_context=shuffle use_kl=0
    C+KL  disc_int_context=true    use_kl=1   lam_init=<lambda_star, required CLI arg>

PHASES (strictly sequential, matching "after all iterations, evaluate 500ep"):
    0. resolve lambda_star (from select_lambda_from_logs.py's output, or --lambda-star)
    1. validate: full 32-way concurrency probe, 8 iterations, before committing
    2. train: all 32 runs, 300 iterations each, 32 workers
    3. final-eval: all 32 runs' saved checkpoints, 500 episodes each, 32 workers
    4. aggregate: per-run JSON (training log-derived eval_history + final 500ep
       eval merged) + cross-seed summary table per condition
    5. git add/commit/push (best-effort, does not fail the run if push fails --
       e.g. no configured remote/credentials on this machine)

Run:
    python run_full_ablation_matrix.py --check-only
    python run_full_ablation_matrix.py --lambda-star 0.01 --validate
    python run_full_ablation_matrix.py --lambda-star 0.01 --workers 32
"""
import argparse, datetime, glob, json, os, re, subprocess, sys, time, uuid
import multiprocessing
import statistics as st

CONDITIONS = ["NC", "C", "SC", "C+KL"]
SEEDS = list(range(8))
ALPHA_MARG = 0.0033
ALPHA_INT = 0.0021
MAX_ITERATIONS = 300
EVAL_EVERY = 50
EVAL_EPISODES = 30
FINAL_EVAL_EPISODES = 500

SEC_PER_ITER_SINGLE = 21.0
SEC_PER_EVAL_EPISODE_SINGLE = 17.5

LAUNCH_ID = f"matrix_{datetime.datetime.now():%m%d}{uuid.uuid4().hex[:4]}"
RESULT_DIR = os.path.join("results_v2", "stage3_full_matrix", LAUNCH_ID)
MANIFEST_PATH = os.path.join("results_v2", "MANIFEST.json")

EVAL_RE = re.compile(
    r"EVAL[^:]*:\s+win=(?P<win>[\d.]+)\s+SAP=(?P<sap>[\d.]+)%\s+"
    r"MECHA=(?P<mecha>[\d.nan]+)\s+CSI=(?P<csi>[-\d.eNone]+)")


def check_resources():
    n_cpu = multiprocessing.cpu_count()
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int([l for l in f if l.startswith("MemAvailable")][0].split()[1])
        mem_gb = mem_kb / 1e6
    except Exception:
        mem_gb = None
    print(f"CPU cores: {n_cpu}")
    print(f"Available RAM: {mem_gb:.1f} GB" if mem_gb else "RAM: could not determine")
    return n_cpu, mem_gb


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def condition_flags(cond_name, lambda_star):
    """Maps the 4 ablation labels onto finetune_loop.py's actual flags."""
    if cond_name == "NC":
        return {"disc_int_context": "zero", "use_kl": "0", "lam_init": "0.0"}
    if cond_name == "C":
        return {"disc_int_context": "true", "use_kl": "0", "lam_init": "0.0"}
    if cond_name == "SC":
        return {"disc_int_context": "shuffle", "use_kl": "0", "lam_init": "0.0"}
    if cond_name == "C+KL":
        return {"disc_int_context": "true", "use_kl": "1", "lam_init": str(lambda_star)}
    raise ValueError(cond_name)


def condition_for(cond_name):
    """Unique per ablation label + launch. Seed provides per-seed
    uniqueness downstream via finetune_loop.py's own
    run_name = f'{condition}_seed{seed}' construction."""
    return f"final-{cond_name}-{LAUNCH_ID}"


def run_name_for(cond_name, seed):
    return f"{condition_for(cond_name)}_seed{seed}"


def build_plan():
    plan = [(c, s) for c in CONDITIONS for s in SEEDS]
    plan.sort(key=lambda x: (x[1], CONDITIONS.index(x[0])))
    return plan


def launch_train_job(cond_name, seed, max_iterations, lambda_star, log_dir):
    flags = condition_flags(cond_name, lambda_star)
    run_name = run_name_for(cond_name, seed)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logf = open(log_path, "w")
    cmd = [sys.executable, "finetune_loop.py",
           "--condition", condition_for(cond_name), "--seed", str(seed),
           "--max-iterations", str(max_iterations),
           "--use-kl", flags["use_kl"], "--lam-init", flags["lam_init"],
           "--no-anneal", "1",
           "--disc-int-context", flags["disc_int_context"],
           "--alpha-marg", str(ALPHA_MARG), "--alpha-int", str(ALPHA_INT),
           "--eval-every", str(EVAL_EVERY), "--eval-episodes", str(EVAL_EPISODES)]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"cond": cond_name, "seed": seed, "run_name": run_name,
            "proc": proc, "logf": logf, "start": time.time()}


def launch_final_eval_job(cond_name, seed, episodes, log_dir):
    run_name = run_name_for(cond_name, seed)
    ckpt_path = f"ablation_{run_name}.pt"
    out_path = os.path.join(RESULT_DIR, f"final_eval_{run_name}.json")
    log_path = os.path.join(log_dir, f"finaleval_{run_name}.log")
    logf = open(log_path, "w")
    cmd = [sys.executable, "evaluate_finetuned_checkpoint.py",
           "--checkpoint", ckpt_path, "--episodes", str(episodes), "--out", out_path]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"cond": cond_name, "seed": seed, "run_name": run_name, "out_path": out_path,
            "proc": proc, "logf": logf, "start": time.time()}


def run_pool(launch_fn, jobs_spec, n_workers, log_dir, label):
    os.makedirs(log_dir, exist_ok=True)
    queue = list(jobs_spec)
    running, completed = [], []
    print(f"\n{'='*70}\n{label}: {len(queue)} jobs, {n_workers}-way concurrency\n{'='*70}")

    while queue or running:
        while queue and len(running) < n_workers:
            args_tuple = queue.pop(0)
            job = launch_fn(*args_tuple, log_dir)
            print(f"  launched {job['run_name']}  [{len(running)+1}/{n_workers} slots, "
                  f"{len(queue)} queued]")
            running.append(job)

        time.sleep(5)
        still = []
        for job in running:
            ret = job["proc"].poll()
            if ret is None:
                still.append(job)
            else:
                job["logf"].close()
                elapsed = time.time() - job["start"]
                print(f"  finished {job['run_name']}  ({elapsed/60:.1f}min, "
                      f"{'OK' if ret == 0 else f'EXIT {ret}'})")
                job["returncode"] = ret
                job["wall_sec"] = elapsed
                completed.append(job)
        running = still

    return completed


def validate(n_workers, log_dir):
    plan = build_plan()[:min(n_workers, len(build_plan()))]
    print(f"\nVALIDATION: {len(plan)} concurrent training jobs, 8 iterations each")
    jobs_spec = [(c, s, 8, 0.01) for (c, s) in plan]   # dummy lambda_star for the probe
    t0 = time.time()
    completed = run_pool(launch_train_job, jobs_spec, n_workers, log_dir, "VALIDATE")
    elapsed = time.time() - t0
    implied = elapsed / 8
    slowdown = implied / SEC_PER_ITER_SINGLE
    print(f"\nWall time: {elapsed:.1f}s  |  implied per-iter cost: {implied:.1f}s  "
          f"(baseline {SEC_PER_ITER_SINGLE}s)  |  slowdown {slowdown:.2f}x")
    ok = all(c["returncode"] == 0 for c in completed)
    if not ok:
        bad = [c["run_name"] for c in completed if c["returncode"] != 0]
        print(f"\nFAIL: non-zero exit from {bad}. Check logs before proceeding.")
        return False, implied
    if slowdown > 1.5:
        print(f"\nWARNING: {slowdown:.2f}x slowdown at {n_workers}-way. Consider fewer workers.")
    else:
        print("\nOK: contention within acceptable range.")
    return True, implied


def parse_training_log(run_name, log_dir):
    path = os.path.join(log_dir, f"{run_name}.log")
    evals = []
    if os.path.exists(path):
        with open(path, errors="ignore") as f:
            for line in f:
                m = EVAL_RE.search(line)
                if m:
                    evals.append({
                        "win_rate": float(m["win"]), "sap_mean": float(m["sap"]),
                        "mecha_mean": (float(m["mecha"]) if m["mecha"] != "nan" else None),
                        "csi_sap_proxy": (float(m["csi"]) if m["csi"] not in ("None", "") else None),
                    })
    return evals


def aggregate_and_write(plan, train_log_dir):
    """Per-run: merge finetune_loop.py's own result_*.json (metadata),
    the training log's eval_history (parsed), and the 500ep final_eval
    JSON into one complete record. Then cross-seed aggregate per condition."""
    per_run = []
    for cond, seed in plan:
        run_name = run_name_for(cond, seed)
        record = {"condition_label": cond, "seed": seed, "run_name": run_name}

        meta_path = f"result_{run_name}.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                record["metadata"] = json.load(f)

        record["eval_history"] = parse_training_log(run_name, train_log_dir)

        final_path = os.path.join(RESULT_DIR, f"final_eval_{run_name}.json")
        if os.path.exists(final_path):
            with open(final_path) as f:
                record["final_eval_500ep"] = json.load(f)
        else:
            record["final_eval_500ep"] = None
            print(f"  WARNING: no final_eval_500ep for {run_name}")

        out_path = os.path.join(RESULT_DIR, f"full_{run_name}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        per_run.append(record)

    print(f"\nwrote {len(per_run)} individual per-run JSON files to {RESULT_DIR}/")

    # cross-seed aggregation per condition, using the 500ep final eval
    # (the directly-comparable, MAPPO-baseline-matched number)
    table = []
    for cond in CONDITIONS:
        rows = [r for r in per_run if r["condition_label"] == cond and r["final_eval_500ep"]]
        wins = [r["final_eval_500ep"]["win_rate"] for r in rows]
        saps = [r["final_eval_500ep"]["sap_mean"] for r in rows]
        mechas = [r["final_eval_500ep"].get("mecha_mean") for r in rows
                 if r["final_eval_500ep"].get("mecha_mean") is not None]
        csis = [r["final_eval_500ep"].get("csi_sap_proxy") for r in rows
               if r["final_eval_500ep"].get("csi_sap_proxy") is not None]

        def summarize(vals):
            if not vals:
                return None
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            se = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
            return {"mean": mean, "sd": sd, "se": se, "n": len(vals)}

        table.append({
            "condition": cond, "n_seeds": len(rows),
            "win_rate": summarize(wins), "sap_mean": summarize(saps),
            "mecha_mean": summarize(mechas), "csi_sap_proxy": summarize(csis),
        })

    summary_path = os.path.join(RESULT_DIR, "ablation_summary_table.json")
    with open(summary_path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"\n{'condition':>8s} {'n':>2s} {'win mean':>9s} {'win se':>7s} "
          f"{'SAP mean':>9s} {'SAP se':>7s}")
    for row in table:
        w, s = row["win_rate"], row["sap_mean"]
        if w and s:
            print(f"{row['condition']:>8s} {row['n_seeds']:>2d} {w['mean']:>9.3f} "
                  f"{w['se']:>7.3f} {s['mean']:>9.2f} {s['se']:>7.2f}")
        else:
            print(f"{row['condition']:>8s}  incomplete data")

    print(f"\nwrote {summary_path}")
    return per_run, table


def git_push():
    print(f"\n{'='*70}\nGIT PUSH\n{'='*70}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        commit = subprocess.run(["git", "commit", "-m", f"full ablation matrix results {LAUNCH_ID}"],
                                capture_output=True, text=True)
        if commit.returncode != 0:
            print(f"  git commit: nothing to commit or failed -- {commit.stdout}{commit.stderr}")
        else:
            print(f"  committed: {commit.stdout.strip()}")
        push = subprocess.run(["git", "push"], capture_output=True, text=True)
        if push.returncode != 0:
            print(f"  git push FAILED (not fatal -- results are saved locally regardless):")
            print(f"  {push.stderr}")
        else:
            print("  pushed successfully.")
    except Exception as e:
        print(f"  git push step raised {e!r} -- not fatal, results are saved locally.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--lambda-star", type=float, default=None,
                    help="required for the real run (not --check-only/--validate)")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    n_cpu, mem_gb = check_resources()
    if args.check_only:
        return

    print(f"\n{len(CONDITIONS)} conditions x {len(SEEDS)} seeds = "
          f"{len(CONDITIONS)*len(SEEDS)} runs  |  workers: {args.workers}")
    if mem_gb is not None and mem_gb < args.workers * 2.0:
        print(f"WARNING: {mem_gb:.1f} GB for {args.workers} workers -- may be tight.")

    train_log_dir = os.path.join(RESULT_DIR, "train_logs")
    eval_log_dir = os.path.join(RESULT_DIR, "eval_logs")

    if args.validate:
        validate(args.workers, train_log_dir)
        return

    if args.lambda_star is None:
        print("\n--lambda-star is required for the real run. Get it from "
              "select_lambda_from_logs.py's output first.")
        return

    if not args.skip_validation:
        ok, _ = validate(args.workers, train_log_dir)
        if not ok:
            print("\nStopping -- validation failed.")
            return

    plan = build_plan()
    os.makedirs(RESULT_DIR, exist_ok=True)

    # PHASE 2: training, all 32 runs, 300 iterations each
    train_jobs_spec = [(c, s, MAX_ITERATIONS, args.lambda_star) for (c, s) in plan]
    t0 = time.time()
    train_completed = run_pool(launch_train_job, train_jobs_spec, args.workers,
                               train_log_dir, "PHASE 2: TRAINING (300 iters)")
    print(f"\nphase 2 done in {(time.time()-t0)/3600:.2f}h  |  "
          f"succeeded: {sum(1 for c in train_completed if c['returncode']==0)}/{len(plan)}")

    # PHASE 3: final 500-episode eval per run, on the SAVED checkpoints
    eval_jobs_spec = [(c, s, FINAL_EVAL_EPISODES) for (c, s) in plan]
    t0 = time.time()
    eval_completed = run_pool(launch_final_eval_job, eval_jobs_spec, args.workers,
                              eval_log_dir, "PHASE 3: FINAL EVAL (500 episodes)")
    print(f"\nphase 3 done in {(time.time()-t0)/3600:.2f}h  |  "
          f"succeeded: {sum(1 for c in eval_completed if c['returncode']==0)}/{len(plan)}")

    # PHASE 4: aggregate
    print(f"\n{'='*70}\nPHASE 4: AGGREGATION\n{'='*70}")
    per_run, table = aggregate_and_write(plan, train_log_dir)

    os.makedirs("results_v2", exist_ok=True)
    manifest = json.load(open(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else []
    manifest.append({
        "launch_id": LAUNCH_ID, "stage": "stage3_full_matrix",
        "utc": datetime.datetime.utcnow().isoformat() + "Z", "git_sha": git_sha(),
        "conditions": CONDITIONS, "seeds": SEEDS, "max_iterations": MAX_ITERATIONS,
        "final_eval_episodes": FINAL_EVAL_EPISODES, "lambda_star": args.lambda_star,
        "alpha_marg": ALPHA_MARG, "alpha_int": ALPHA_INT,
    })
    json.dump(manifest, open(MANIFEST_PATH, "w"), indent=2)
    print(f"appended to {MANIFEST_PATH}")

    # PHASE 5: auto git push (best-effort, after EVERYTHING above)
    git_push()

    print(f"\n{'='*70}\nALL PHASES COMPLETE -- {RESULT_DIR}/\n{'='*70}")


if __name__ == "__main__":
    main()
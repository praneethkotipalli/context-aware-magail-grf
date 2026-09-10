"""
run_lambda_sweep.py -- FIXED.

Three bugs in the version this replaces:
  1. CRITICAL, SILENT: --condition was hardcoded to "MAGAIL-C+KL" for
     every job. finetune_loop.py builds run_name = f"{condition}_seed{seed}"
     internally, so all 5 lambdas at seed=0 collided on the SAME
     result_*.json / ablation_*.pt / wandb run name -- and since
     build_plan() sorts by seed first, those 5 jobs run CONCURRENTLY,
     meaning multiple processes write to the same three paths while all
     still running. No crash, no error -- just silently corrupted output.
     FIX: condition now encodes lambda + LAUNCH_ID, made unique per job.
  2. --run-name-override does not exist on finetune_loop.py's CLI. REMOVED.
  3. --wandb-project does not exist either (only as a run() kwarg,
     defaulting to "magail-c-v2" already). REMOVED, nothing lost.
  4. --no-anneal was passed bare; it's type=int not store_true. FIX:
     pass "--no-anneal", "1" explicitly.

Everything else (validation discipline, storage layout, lambda* selection
rule, MANIFEST append) is unchanged.

Run:
    python run_lambda_sweep.py --check-only
    python run_lambda_sweep.py --validate
    python run_lambda_sweep.py --workers 8 --max-iterations 150
"""

import argparse, datetime, json, os, subprocess, sys, time, uuid
import multiprocessing

LAMBDAS = [0.0, 0.003, 0.01, 0.03, 0.1]
SEEDS = [0, 1, 2]
BASELINE_WIN_RATE = 0.624
SAP_MOVEMENT_FLOOR = 5.0

ALPHA_MARG = 0.0033
ALPHA_INT = 0.0021

SEC_PER_ITER_SINGLE = 21.0
SEC_PER_EVAL_EPISODE_SINGLE = 17.5

LAUNCH_ID = f"lsweep_{datetime.datetime.now():%m%d}{uuid.uuid4().hex[:4]}"
RESULT_DIR = os.path.join("results_v2", "stage2_lambda_sweep", LAUNCH_ID)
MANIFEST_PATH = os.path.join("results_v2", "MANIFEST.json")
WANDB_PROJECT = "magail-c-v2"   # matches finetune_loop.py's run() default -- not passed on CLI


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
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def condition_for(lam):
    """[FIX 1] unique per lambda + launch, so run_name never collides."""
    return f"lsweep-lam{lam:g}-{LAUNCH_ID}"


def run_name_for(lam, seed):
    """Mirrors finetune_loop.py's OWN internal construction exactly:
    run_name = f'{condition}_seed{seed}'. Computed here, not stored
    separately, so it can never drift out of sync with what the CLI
    subprocess actually writes to disk."""
    return f"{condition_for(lam)}_seed{seed}"


def build_plan():
    """Each entry: (lam, seed). condition/run_name derived on demand via
    condition_for()/run_name_for() -- single source of truth."""
    plan = [(lam, seed) for lam in LAMBDAS for seed in SEEDS]
    plan.sort(key=lambda x: (x[1], LAMBDAS.index(x[0])))
    return plan


def launch_job(lam, seed, max_iterations, eval_every, eval_episodes, log_dir):
    condition = condition_for(lam)
    run_name = run_name_for(lam, seed)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logf = open(log_path, "w")
    cmd = [
        sys.executable, "finetune_loop.py",
        "--condition", condition,                    # [FIX 1] unique, not hardcoded
        "--seed", str(seed),
        "--max-iterations", str(max_iterations),
        "--use-kl", "1",
        "--lam-init", str(lam),
        "--no-anneal", "1",                           # [FIX 4] explicit value, type=int
        "--alpha-marg", str(ALPHA_MARG),
        "--alpha-int", str(ALPHA_INT),
        "--disc-int-context", "true",
        "--eval-every", str(eval_every),
        "--eval-episodes", str(eval_episodes),
        # [FIX 2, 3] --run-name-override and --wandb-project REMOVED --
        # neither exists on finetune_loop.py's CLI.
    ]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"lam": lam, "seed": seed, "condition": condition, "run_name": run_name,
            "proc": proc, "logf": logf, "start": time.time()}


def run_pool(jobs_spec, n_workers, eval_every, eval_episodes, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    queue = list(jobs_spec)
    running, completed = [], []

    while queue or running:
        while queue and len(running) < n_workers:
            lam, seed, n_it = queue.pop(0)
            job = launch_job(lam, seed, n_it, eval_every, eval_episodes, log_dir)
            print(f"  launched {job['run_name']}  (lam={lam:g}, seed={seed})  "
                  f"[{len(running)+1}/{n_workers} slots, {len(queue)} queued]")
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
                completed.append({"lam": job["lam"], "seed": job["seed"],
                                  "run_name": job["run_name"], "returncode": ret,
                                  "wall_sec": elapsed})
        running = still

    return completed


def validate(n_workers_to_test, log_dir="results_v2/stage2_lambda_sweep/validation"):
    plan = build_plan()[:n_workers_to_test]
    print(f"\n{'='*70}\nVALIDATION: {len(plan)} concurrent lambda-sweep jobs, "
          f"8 iterations each\n{'='*70}")
    jobs_spec = [(lam, seed, 8) for (lam, seed) in plan]

    t0 = time.time()
    completed = run_pool(jobs_spec, n_workers_to_test, eval_every=1000,
                         eval_episodes=1, log_dir=log_dir)
    elapsed = time.time() - t0

    n_iters_each = 8
    implied = elapsed / n_iters_each
    slowdown = implied / SEC_PER_ITER_SINGLE
    print(f"\nWall time: {elapsed:.1f}s  |  implied per-iteration cost: {implied:.1f}s  "
          f"(single-worker baseline {SEC_PER_ITER_SINGLE:.1f}s)  |  slowdown {slowdown:.2f}x")

    ok = all(c["returncode"] == 0 for c in completed)
    if not ok:
        bad = [c["run_name"] for c in completed if c["returncode"] != 0]
        print(f"\nFAIL: non-zero exit from {bad}. Check logs in {log_dir}/.")
        return False, implied
    if slowdown > 1.5:
        print(f"\nWARNING: {slowdown:.2f}x slowdown at {n_workers_to_test}-way concurrency.")
    else:
        print("\nOK: contention within acceptable range.")
    return True, implied


def solve_iterations(budget_sec, n_batches, eval_every, eval_episodes,
                     sec_per_iter, safety=0.85):
    per_batch_budget = budget_sec * safety / n_batches
    lo, hi = 10, 1000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n_evals = mid // eval_every
        est = mid * sec_per_iter + n_evals * eval_episodes * SEC_PER_EVAL_EPISODE_SINGLE
        if est <= per_batch_budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def collect_results(plan, log_dir):
    import re
    EVAL_RE = re.compile(
        r"EVAL[^:]*:\s+win=(?P<win>[\d.]+)\s+SAP=(?P<sap>[\d.]+)%\s+"
        r"MECHA=(?P<mecha>[\d.nan]+)\s+CSI=(?P<csi>[-\d.eNone]+)")
    results = []
    for lam, seed in plan:
        run_name = run_name_for(lam, seed)
        json_path = f"result_{run_name}.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                r = json.load(f)
            r.setdefault("lam", lam); r.setdefault("seed", seed)
            results.append(r)
            continue
        log_path = os.path.join(log_dir, f"{run_name}.log")
        parsed = {"lam": lam, "seed": seed, "run_name": run_name,
                  "source": "log_fallback", "eval_history": []}
        if os.path.exists(log_path):
            with open(log_path, errors="ignore") as f:
                for line in f:
                    m = EVAL_RE.search(line)
                    if m:
                        parsed["eval_history"].append({
                            "win_rate": float(m["win"]), "sap_mean": float(m["sap"]),
                            "mecha_mean": (float(m["mecha"]) if m["mecha"] != "nan" else None),
                            "csi_sap_proxy": (float(m["csi"]) if m["csi"] not in ("None", "") else None),
                        })
            if parsed["eval_history"]:
                parsed["final_eval"] = parsed["eval_history"][-1]
        results.append(parsed)
    return results


def select_lambda_star(results):
    import statistics as st
    by_lam = {}
    for r in results:
        fe = r.get("final_eval") or (r.get("eval_history") or [{}])[-1]
        wr, sap = fe.get("win_rate"), fe.get("sap_mean")
        if wr is None:
            continue
        by_lam.setdefault(r["lam"], {"win": [], "sap": []})
        by_lam[r["lam"]]["win"].append(wr)
        if sap is not None:
            by_lam[r["lam"]]["sap"].append(sap)

    print(f"\n{'lambda':>8s} {'n':>2s} {'mean win':>10s} {'win SE':>8s} "
          f"{'mean SAP':>9s} {'moved?':>7s} {'eligible?':>10s}")
    candidates = []
    for lam in sorted(by_lam):
        wins = by_lam[lam]["win"]; saps = by_lam[lam]["sap"]
        mean_win = st.mean(wins)
        se_win = (st.stdev(wins) / len(wins) ** 0.5) if len(wins) > 1 else float("nan")
        mean_sap = st.mean(saps) if saps else float("nan")
        moved = (85.9 - mean_sap) >= SAP_MOVEMENT_FLOOR if saps else False
        eligible = mean_win >= (BASELINE_WIN_RATE - (se_win if se_win == se_win else 0))
        print(f"{lam:>8.3g} {len(wins):>2d} {mean_win:>10.3f} {se_win:>8.3f} "
              f"{mean_sap:>9.2f} {'yes' if moved else 'no':>7s} "
              f"{'yes' if eligible and moved else 'no':>10s}")
        if eligible and moved:
            candidates.append(lam)

    if not candidates:
        print("\nNo lambda is both win-preserving AND shows real SAP movement.")
        return None
    lam_star = min(candidates)
    print(f"\nlambda* = {lam_star:g}")
    return lam_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget-hours", type=float, default=7.0)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    n_cpu, mem_gb = check_resources()
    if args.check_only:
        return

    n_workers = args.workers or min(len(LAMBDAS) * len(SEEDS), max(1, n_cpu - 1))
    print(f"\nProposed workers: {n_workers} (of {n_cpu} cores)  |  "
          f"{len(LAMBDAS)*len(SEEDS)} total runs")
    if mem_gb is not None and mem_gb < n_workers * 2.0:
        print(f"WARNING: {mem_gb:.1f} GB available for {n_workers} workers.")

    plan = build_plan()

    if args.validate:
        validate(min(n_workers, len(plan)))
        return

    measured_sec_per_iter = SEC_PER_ITER_SINGLE
    if not args.skip_validation:
        ok, implied = validate(min(n_workers, len(plan)))
        if not ok:
            print("\nStopping -- validation failed.")
            return
        measured_sec_per_iter = implied

    n_batches = -(-len(plan) // n_workers)
    n_iters = args.max_iterations or solve_iterations(
        args.budget_hours * 3600, n_batches, args.eval_every, args.eval_episodes,
        measured_sec_per_iter)
    est_hours = (n_iters * measured_sec_per_iter
                + (n_iters // args.eval_every) * args.eval_episodes * SEC_PER_EVAL_EPISODE_SINGLE
                ) * n_batches / 3600

    print(f"\n{n_workers}-way concurrency  |  {len(plan)} runs  |  {n_batches} batches  |  "
          f"iterations/run: {n_iters}  |  estimated total: {est_hours:.1f}h")

    os.makedirs(RESULT_DIR, exist_ok=True)
    log_dir = os.path.join(RESULT_DIR, "logs")
    jobs_spec = [(lam, seed, n_iters) for (lam, seed) in plan]

    t_start = time.time()
    completed = run_pool(jobs_spec, n_workers, args.eval_every, args.eval_episodes, log_dir)
    total = time.time() - t_start

    print(f"\n{'='*70}\nALL DONE: {total/3600:.2f}h for {len(plan)} runs\n{'='*70}")
    ok_n = sum(1 for c in completed if c["returncode"] == 0)
    print(f"Succeeded: {ok_n}/{len(plan)}")

    results = collect_results(plan, log_dir)
    with open(os.path.join(RESULT_DIR, "lambda_sweep_full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULT_DIR}/lambda_sweep_full_results.json")

    lam_star = select_lambda_star(results)
    with open(os.path.join(RESULT_DIR, "lambda_star.json"), "w") as f:
        json.dump({"lambda_star": lam_star, "rule": "smallest eligible+moved lambda, "
                  "win >= baseline-1SE, SAP moved >= floor",
                  "sap_movement_floor": SAP_MOVEMENT_FLOOR,
                  "baseline_win_rate": BASELINE_WIN_RATE, "launch_id": LAUNCH_ID}, f, indent=2)

    os.makedirs("results_v2", exist_ok=True)
    manifest_entry = {
        "launch_id": LAUNCH_ID, "stage": "stage2_lambda_sweep",
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha(), "wandb_project": WANDB_PROJECT,
        "lambdas": LAMBDAS, "seeds": SEEDS, "n_runs": len(plan),
        "max_iterations": n_iters, "alpha_marg": ALPHA_MARG, "alpha_int": ALPHA_INT,
        "lambda_star": lam_star, "succeeded": ok_n,
    }
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    manifest.append(manifest_entry)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"appended to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
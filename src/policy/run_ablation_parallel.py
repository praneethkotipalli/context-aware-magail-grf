"""
run_ablation_parallel.py

Runs the 15 ablation jobs as SEPARATE OS PROCESSES (subprocess.Popen, not
threads/fork) up to N concurrently, where N is chosen from real core count
-- checked, not assumed.

MANDATORY VALIDATION PHASE, cannot be skipped via a flag: launches 2 jobs
concurrently at a small iteration count FIRST, measures whether per-worker
time degrades vs. the known single-worker baseline (~21s/iteration,
measured previously on this exact machine). Only proceeds to the full
15-job batch if validation shows acceptable overhead. This exists because
the alternative -- finding out 3 hours into a 9-hour unattended run that
concurrent GRF instances contend badly for CPU/memory -- is much worse
than spending 5 minutes checking first.

Usage:
    python run_ablation_parallel.py --check-only        # just print core/RAM info
    python run_ablation_parallel.py --validate           # 2-worker contention test only
    python run_ablation_parallel.py --workers 4 --budget-hours 9   # full run, skips
                                                                     # validation ONLY if
                                                                     # --skip-validation also passed
"""

import argparse, json, os, subprocess, sys, time
import multiprocessing

SEC_PER_ITER_SINGLE = 21.0          # measured baseline, single worker, this machine
SEC_PER_EVAL_EPISODE_SINGLE = 17.5
ACCEPTABLE_SLOWDOWN = 1.5            # if concurrent per-iter time exceeds 1.5x the
                                       # single-worker baseline, contention is bad enough
                                       # that concurrency isn't worth the complexity

SEEDS = [0, 1, 2, 3, 4]
SEEDS_SWEEP = [0, 1, 2]
LAMBDA_SWEEP = [0.001, 0.01, 0.1, 1.0]

# Each entry: (condition_name, use_context, shuffle_context, use_kl, frozen, lam)
# lam=None -> finetune_objective's default (1.0) is used.
CORE_CONDITIONS = [
    ("MAGAIL-C",     True,  False, False, False, None),
    ("MAGAIL-SC",    True,  True,  False, False, None),
    ("MAGAIL-C+KL",  True,  False, True,  False, None),
]
EXTRA_CONDITIONS_FOR_FULL = [
    ("MAPPO-baseline", False, False, False, True,  None),
    ("MAGAIL-NC",      False, False, False, False, None),
]


def check_resources():
    n_cpu = multiprocessing.cpu_count()
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int([l for l in f if l.startswith("MemAvailable")][0].split()[1])
        mem_gb = mem_kb / 1e6
    except Exception:
        mem_gb = None
    print(f"CPU cores: {n_cpu}")
    print(f"Available RAM: {mem_gb:.1f} GB" if mem_gb else "Available RAM: could not determine -- check `free -h` manually")
    return n_cpu, mem_gb


def build_plan(full=False):
    """
    full=False (default, already validated tonight): 15 jobs -- MAGAIL-C,
    MAGAIL-SC, MAGAIL-C+KL x 5 seeds.

    full=True: all 37 -- adds MAPPO-baseline (eval-only, cheap), MAGAIL-NC
    (5 seeds), and the lambda sweep (4 values x 3 seeds). NOT validated at
    this concurrency (only 15-way tested) -- run --validate again at the
    real 37-way scale before trusting the timing math for it.

    Each plan entry: (condition, seed, use_context, shuffle_context,
    use_kl, frozen, lam).
    """
    plan = [(name, s, uc, sc, kl, fr, lam)
            for (name, uc, sc, kl, fr, lam) in CORE_CONDITIONS for s in SEEDS]

    if full:
        plan += [(name, s, uc, sc, kl, fr, lam)
                 for (name, uc, sc, kl, fr, lam) in EXTRA_CONDITIONS_FOR_FULL for s in SEEDS]
        for lam in LAMBDA_SWEEP:
            for s in SEEDS_SWEEP:
                plan.append((f"lambda_sweep_{lam}", s, True, False, True, False, lam))

    # priority order preserved: core conditions first (interleaved by
    # seed), then baseline/NC, then the sweep last -- matches the
    # already-agreed triage (sweep is only ever a "trend estimate")
    name_order = ([c[0] for c in CORE_CONDITIONS]
                  + [c[0] for c in EXTRA_CONDITIONS_FOR_FULL]
                  + [f"lambda_sweep_{l}" for l in LAMBDA_SWEEP])
    order = {n: i for i, n in enumerate(name_order)}
    plan.sort(key=lambda x: (x[1], order.get(x[0], 99)))
    return plan


def launch_job(condition, seed, use_ctx, shuf, kl, frozen, lam, max_iters, eval_every, eval_episodes, log_dir):
    run_name = f"{condition}_seed{seed}"
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logf = open(log_path, "w")
    cmd = [
        sys.executable, "finetune_loop.py",
        "--condition", condition, "--seed", str(seed),
        "--max-iterations", str(max_iters),
        "--use-kl", "1" if kl else "0",
        "--shuffle-context", "1" if shuf else "0",
        "--use-context", "1" if use_ctx else "0",
        "--eval-every", str(eval_every), "--eval-episodes", str(eval_episodes),
        "--frozen", "1" if frozen else "0",
    ]
    if lam is not None:
        cmd += ["--lam-init", str(lam)]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"run_name": run_name, "proc": proc, "logf": logf, "start": time.time()}


def run_pool(jobs_spec, n_workers, eval_every, eval_episodes, log_dir):
    """jobs_spec: list of (condition, seed, use_ctx, shuf, kl, frozen, lam, max_iters)."""
    os.makedirs(log_dir, exist_ok=True)
    queue = list(jobs_spec)
    running = []
    completed = []

    while queue or running:
        while queue and len(running) < n_workers:
            c, s, uc, sc, kl, fr, lam, n_it = queue.pop(0)
            job = launch_job(c, s, uc, sc, kl, fr, lam, n_it, eval_every, eval_episodes, log_dir)
            print(f"  launched {job['run_name']}  ({len(running)+1}/{n_workers} slots, "
                  f"{len(queue)} still queued)")
            running.append(job)

        time.sleep(5)
        still_running = []
        for job in running:
            ret = job["proc"].poll()
            if ret is None:
                still_running.append(job)
            else:
                job["logf"].close()
                elapsed = time.time() - job["start"]
                status = "OK" if ret == 0 else f"EXIT CODE {ret}"
                print(f"  finished {job['run_name']}  ({elapsed/60:.1f}min, {status})")
                completed.append({"run_name": job["run_name"], "returncode": ret, "wall_sec": elapsed})
        running = still_running

    return completed


def validate(n_workers_to_test, full=False, log_dir="ablation_logs_validation"):
    """Runs n_workers_to_test REAL jobs (from the actual plan, not fake
    duplicates) concurrently at a small iteration count, comparing measured
    per-iteration time against the single-worker baseline."""
    plan = [p for p in build_plan(full=full) if not p[5]][:n_workers_to_test]  # skip frozen (no training to time)
    print(f"\n{'='*70}\nVALIDATION: {len(plan)} concurrent REAL jobs, 15 iterations each\n{'='*70}")
    test_jobs = [(c, s, uc, sc, kl, fr, lam, 15) for (c, s, uc, sc, kl, fr, lam) in plan]

    t0 = time.time()
    completed = run_pool(test_jobs, n_workers=len(plan),
                         eval_every=1000, eval_episodes=1, log_dir=log_dir)  # eval disabled for a clean timing read
    elapsed = time.time() - t0

    n_iters_each = 15
    implied_per_iter = elapsed / n_iters_each
    slowdown = implied_per_iter / SEC_PER_ITER_SINGLE

    print(f"\nWall time for {len(plan)} concurrent 15-iteration jobs: {elapsed:.1f}s")
    print(f"Implied per-iteration time under contention: {implied_per_iter:.1f}s "
          f"(single-worker baseline: {SEC_PER_ITER_SINGLE:.1f}s)")
    print(f"Slowdown factor: {slowdown:.2f}x")

    all_ok = all(c["returncode"] == 0 for c in completed)
    if not all_ok:
        print("\nFAIL: at least one validation job exited with an error -- check logs in "
              f"{log_dir}/ before proceeding. DO NOT run the full batch yet.")
        return False
    if slowdown > ACCEPTABLE_SLOWDOWN:
        print(f"\nFAIL: slowdown {slowdown:.2f}x exceeds the {ACCEPTABLE_SLOWDOWN}x threshold. "
              f"Concurrent execution is contending badly (CPU or memory) -- running the full "
              f"batch this way would likely take LONGER than sequential, not less. "
              f"Consider fewer concurrent workers or falling back to sequential (run_ablation.py).")
        return False

    print(f"\nOK: slowdown within acceptable range. Safe to proceed with "
          f"{n_workers_to_test}-way concurrency for the full batch.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget-hours", type=float, default=9.0)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--skip-validation", action="store_true",
                    help="DANGEROUS on an unattended long run -- only use if you already "
                         "validated this exact worker count recently")
    ap.add_argument("--full", action="store_true",
                    help="run all 37 jobs (adds MAPPO-baseline, MAGAIL-NC, lambda sweep) "
                         "instead of the 15 already validated tonight -- NOT validated at "
                         "this concurrency, validation will re-run automatically at the real scale")
    ap.add_argument("--iterations", type=int, default=None,
                    help="OVERRIDE the auto-computed iteration count. Use this if you already "
                         "have a real measured per-iteration time from --validate at this exact "
                         "worker count -- the auto-solver still uses the ORIGINAL single-worker "
                         "baseline (21.0s), which underestimates cost once real contention is "
                         "measured at higher concurrency (e.g. 28.7s measured at 37-way tonight).")
    args = ap.parse_args()

    n_cpu, mem_gb = check_resources()
    if args.check_only:
        return

    plan_full = build_plan(full=args.full)
    n_trainable = len([p for p in plan_full if not p[5]])   # excludes frozen baseline (no training cost)
    n_workers = args.workers or min(len(plan_full), max(1, n_cpu - 1))
    print(f"\nProposed concurrent workers: {n_workers} (of {n_cpu} cores)  |  "
          f"{len(plan_full)} total jobs ({n_trainable} trained, {len(plan_full)-n_trainable} frozen-eval-only)")
    if mem_gb is not None and mem_gb < n_workers * 2.0:
        print(f"WARNING: {mem_gb:.1f} GB available, {n_workers} workers requested. "
              f"Each GRF instance can use noticeable memory -- if this is too tight, "
              f"reduce --workers rather than find out via an OOM kill mid-run.")

    if args.validate:
        validate(min(n_workers, n_trainable), full=args.full)
        return

    if not args.skip_validation:
        ok = validate(min(n_workers, n_trainable), full=args.full)
        if not ok:
            print("\nStopping -- validation did not pass. Fix the issue or pass "
                  "--skip-validation to override (not recommended).")
            return

    plan = plan_full
    budget_sec = args.budget_hours * 3600
    n_batches = -(-len(plan) // n_workers)  # ceil
    per_run_budget = (budget_sec * 0.85) / n_batches

    if args.iterations is not None:
        n_iters = args.iterations
        print(f"\nUsing MANUAL iteration override: {n_iters} (bypassing auto-solver)")
    else:
        lo, hi = 10, 3000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            n_evals = mid // args.eval_every
            est = mid * SEC_PER_ITER_SINGLE + n_evals * args.eval_episodes * SEC_PER_EVAL_EPISODE_SINGLE
            if est <= per_run_budget:
                lo = mid
            else:
                hi = mid - 1
        n_iters = lo
        print(f"\nWARNING: using auto-solved iterations based on the {SEC_PER_ITER_SINGLE}s/iter "
              f"CONSTANT, not any real measurement from your last --validate run. If you have a "
              f"real number, pass --iterations explicitly instead of trusting this.")

    print(f"\n{n_workers}-way concurrency  |  {len(plan)} runs  |  {n_batches} batches  |  "
          f"iterations/run: {n_iters}")

    jobs_spec = [(c, s, uc, sc, kl, fr, lam, (1 if fr else n_iters)) for (c, s, uc, sc, kl, fr, lam) in plan]
    log_dir = "ablation_logs"
    t_start = time.time()
    completed = run_pool(jobs_spec, n_workers, args.eval_every, args.eval_episodes, log_dir)
    total = time.time() - t_start

    print(f"\n{'='*70}\nALL DONE: {total/3600:.2f}h wall time for {len(plan)} runs\n{'='*70}")
    ok = [c for c in completed if c["returncode"] == 0]
    bad = [c for c in completed if c["returncode"] != 0]
    print(f"Succeeded: {len(ok)}/{len(plan)}")
    if bad:
        print(f"FAILED (check logs in {log_dir}/): {[c['run_name'] for c in bad]}")

    with open("ablation_parallel_summary.json", "w") as f:
        json.dump(completed, f, indent=2)


if __name__ == '__main__':
    main()
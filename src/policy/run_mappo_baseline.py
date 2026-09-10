"""
run_mappo_baseline.py -- FIXED.

Two bugs in the version this replaces:
  1. --run-name-override and --wandb-project don't exist on
     finetune_loop.py's CLI (same category of bug as the lambda sweep,
     confirmed against the actual source). REMOVED.
  2. validate() requested 10 episodes, but finetune_loop.py's frozen
     branch enforces n_episodes=max(eval_episodes, 50) internally -- 50
     always actually ran, silently, while implied-cost math divided by
     10. FIXED: validate() now requests 50 explicitly, matching the
     floor, so requested == actual and the math is correct.

No run_name collision risk here (unlike the lambda sweep): --condition
is fixed at "MAPPO-baseline" for every job, and finetune_loop.py builds
run_name = f"{condition}_seed{seed}" internally -- seed alone already
makes every job's name unique. run_name_for() below just needs to
PREDICT that exact string correctly, not manufacture uniqueness.

Run:
    python run_mappo_baseline.py --check-only
    python run_mappo_baseline.py --validate
    python run_mappo_baseline.py --workers 8 --episodes 500
"""

import argparse, datetime, json, os, subprocess, sys, time, uuid
import multiprocessing

SEEDS = list(range(8))
EPISODES = 500
EVAL_FLOOR = 50   # [FIX 2] finetune_loop.py's frozen branch: max(eval_episodes, 50)

LAUNCH_ID = f"baseline_{datetime.datetime.now():%m%d}{uuid.uuid4().hex[:4]}"
RESULT_DIR = os.path.join("results_v2", "stage3_baseline", LAUNCH_ID)
MANIFEST_PATH = os.path.join("results_v2", "MANIFEST.json")

SEC_PER_EPISODE_SINGLE = 17.5


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


def run_name_for(seed):
    """[FIX] The EXACT string finetune_loop.py builds internally:
    run_name = f'{condition}_seed{seed}' with condition fixed at
    'MAPPO-baseline'. Used for log filenames AND result-json lookup --
    single source of truth, same principle as the lambda sweep fix."""
    return f"MAPPO-baseline_seed{seed}"


def launch_job(seed, episodes, log_dir):
    run_name = run_name_for(seed)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    logf = open(log_path, "w")
    cmd = [
        sys.executable, "finetune_loop.py",
        "--condition", "MAPPO-baseline",
        "--seed", str(seed),
        "--max-iterations", "1",           # unused in frozen mode, required arg regardless
        "--frozen", "1",
        "--eval-episodes", str(episodes),
        # [FIX 1] --run-name-override and --wandb-project REMOVED --
        # neither exists on finetune_loop.py's CLI.
    ]
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return {"seed": seed, "run_name": run_name, "proc": proc, "logf": logf, "start": time.time()}


def run_pool(jobs_spec, n_workers, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    queue = list(jobs_spec)
    running, completed = [], []

    while queue or running:
        while queue and len(running) < n_workers:
            seed, episodes = queue.pop(0)
            job = launch_job(seed, episodes, log_dir)
            print(f"  launched {job['run_name']}  (seed={seed})  "
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
                completed.append({"seed": job["seed"], "run_name": job["run_name"],
                                  "returncode": ret, "wall_sec": elapsed})
        running = still

    return completed


def validate(n_workers_to_test, log_dir="results_v2/stage3_baseline/validation"):
    print(f"\n{'='*70}\nVALIDATION: {n_workers_to_test} concurrent baseline jobs, "
          f"{EVAL_FLOOR} episodes each (matches finetune_loop.py's enforced floor)\n{'='*70}")
    jobs_spec = [(s, EVAL_FLOOR) for s in SEEDS[:n_workers_to_test]]  # [FIX 2]

    t0 = time.time()
    completed = run_pool(jobs_spec, n_workers_to_test, log_dir)
    elapsed = time.time() - t0

    implied = elapsed / EVAL_FLOOR
    slowdown = implied / SEC_PER_EPISODE_SINGLE
    print(f"\nWall time: {elapsed:.1f}s  |  implied per-episode cost: {implied:.2f}s  "
          f"(single-worker reference {SEC_PER_EPISODE_SINGLE}s)  |  slowdown {slowdown:.2f}x")

    ok = all(c["returncode"] == 0 for c in completed)
    if not ok:
        bad = [c["run_name"] for c in completed if c["returncode"] != 0]
        print(f"\nFAIL: non-zero exit from {bad}. Check logs in {log_dir}/.")
        return False, implied
    if slowdown > 1.5:
        print(f"\nWARNING: {slowdown:.2f}x slowdown. Running alongside the 15-worker "
              f"lambda sweep may be contending harder than expected -- consider "
              f"--workers 4-6 instead of 8 if this holds.")
    else:
        print("\nOK: contention within acceptable range alongside the sweep.")
    return True, implied


def collect_results(seeds, log_dir):
    import re
    EVAL_RE = re.compile(
        r"win=(?P<win>[\d.]+)\s+SAP=(?P<sap>[\d.]+)%\s+"
        r"MECHA=(?P<mecha>[\d.nan]+)\s+CSI=(?P<csi>[-\d.eNone]+)")
    results = []
    for seed in seeds:
        run_name = run_name_for(seed)
        json_path = f"result_{run_name}.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                r = json.load(f)
            r.setdefault("seed", seed)
            results.append(r)
            continue
        log_path = os.path.join(log_dir, f"{run_name}.log")
        parsed = {"seed": seed, "run_name": run_name, "source": "log_fallback"}
        if os.path.exists(log_path):
            with open(log_path, errors="ignore") as f:
                text = f.read()
            m = None
            for m in EVAL_RE.finditer(text):
                pass
            if m:
                parsed.update({
                    "win_rate": float(m["win"]), "sap_mean": float(m["sap"]),
                    "mecha_mean": (float(m["mecha"]) if m["mecha"] != "nan" else None),
                    "csi_sap_proxy": (float(m["csi"]) if m["csi"] not in ("None", "") else None),
                })
        results.append(parsed)
    return results


def summarise(results):
    import statistics as st
    wins = [r["win_rate"] for r in results if r.get("win_rate") is not None]
    saps = [r["sap_mean"] for r in results if r.get("sap_mean") is not None]
    csis = [r["csi_sap_proxy"] for r in results if r.get("csi_sap_proxy") is not None]

    def stats(vals, name):
        if not vals:
            print(f"  {name}: no data"); return None
        mean, sd = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
        se = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
        print(f"  {name}: mean={mean:.4f}  sd={sd:.4f}  se={se:.4f}  n={len(vals)}  "
              f"values={[round(v,4) for v in vals]}")
        return {"mean": mean, "sd": sd, "se": se, "n": len(vals), "values": vals}

    print("\n" + "=" * 70)
    print(f"MAPPO-BASELINE SUMMARY  ({len(results)} seeds)")
    print("=" * 70)
    summary = {
        "win_rate": stats(wins, "win_rate"),
        "sap_mean": stats(saps, "sap_mean"),
        "csi_sap_proxy": stats(csis, "csi_sap_proxy"),
    }
    if summary["win_rate"]:
        print(f"\nUpdate BASELINE_WIN_RATE in downstream scripts to: "
              f"{summary['win_rate']['mean']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    n_cpu, mem_gb = check_resources()
    if args.check_only:
        return

    print(f"\nWorkers requested: {args.workers}  |  seeds: {len(SEEDS)}  |  "
          f"episodes/seed: {args.episodes}")
    if mem_gb is not None and mem_gb < args.workers * 2.0:
        print(f"WARNING: {mem_gb:.1f} GB available for {args.workers} workers.")

    if args.validate:
        validate(min(args.workers, len(SEEDS)))
        return

    if not args.skip_validation:
        ok, implied = validate(min(args.workers, len(SEEDS)))
        if not ok:
            print("\nStopping -- validation failed.")
            return
        est_min = (implied * args.episodes) / 60
        print(f"\nEstimated wall time at {implied:.2f}s/episode: {est_min:.1f} min "
              f"(all 8 seeds run concurrently)")

    os.makedirs(RESULT_DIR, exist_ok=True)
    log_dir = os.path.join(RESULT_DIR, "logs")
    jobs_spec = [(s, args.episodes) for s in SEEDS]

    t_start = time.time()
    completed = run_pool(jobs_spec, args.workers, log_dir)
    total = time.time() - t_start

    print(f"\n{'='*70}\nALL DONE: {total/60:.1f}min for {len(SEEDS)} seeds\n{'='*70}")
    ok_n = sum(1 for c in completed if c["returncode"] == 0)
    print(f"Succeeded: {ok_n}/{len(SEEDS)}")

    results = collect_results(SEEDS, log_dir)
    with open(os.path.join(RESULT_DIR, "baseline_full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULT_DIR}/baseline_full_results.json")

    summary = summarise(results)
    with open(os.path.join(RESULT_DIR, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    os.makedirs("results_v2", exist_ok=True)
    manifest_entry = {
        "launch_id": LAUNCH_ID, "stage": "stage3_baseline",
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha(), "seeds": SEEDS, "episodes_per_seed": args.episodes,
        "succeeded": ok_n,
        "win_rate_mean": summary["win_rate"]["mean"] if summary["win_rate"] else None,
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
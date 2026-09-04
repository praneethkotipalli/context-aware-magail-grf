"""
run_ablation.py -- time-budgeted ablation driver.

Usage:
    python run_ablation.py --budget-hours 9 --dry-run
    python run_ablation.py --budget-hours 9

Given a hard wall-clock budget, computes how many iterations per run are
affordable, prints the plan and ETA, then runs sequentially with W&B
logging per run. Writes ablation_progress.json after every run so an
interrupted session can be resumed rather than restarted.

SCOPE, given the 10-hour window before forced restart:
  MAGAIL-C  (5 seeds)  -- headline context condition
  MAGAIL-SC (5 seeds)  -- causal control, the actual novelty claim
  MAGAIL-C+KL (5 seeds)-- full system, lam=1.0 (locked initial value)
  MAPPO baseline -- NOT trained; evaluate the frozen policy directly
      (cheap, no gradient steps). Run separately via evaluate_policy.
  MAGAIL-NC and the lambda sweep -- DEFERRED to Monday, per the
      already-agreed triage order.

KNOWN LIMITATION, stated not hidden -- MAGAIL-SC:
  Section 3.5.2 defines SC as context vectors "randomly permuted across
  transitions during discriminator training". This implementation
  permutes context during the ONLINE discriminator updates, but all
  conditions still START from the SAME gate-passing Phase B checkpoint,
  which was pre-trained with TRUE context. A fully clean SC would also
  re-run Phase A/B pre-training with shuffled context (~30-40 min).
  Consequence: the SC condition here is "true-context pre-training +
  shuffled-context online updates", which is a WEAKER control than the
  methodology specifies -- it understates rather than overstates any
  C-vs-SC difference, but it must be reported accurately as implemented.
  If time allows before Monday, pre-train a shuffled-context
  discriminator and re-run SC properly.
"""

import argparse, json, os, time

CONDITIONS = [
    # (name, use_context, shuffle_context, use_kl)
    ("MAGAIL-C",    True,  False, False),
    ("MAGAIL-SC",   True,  True,  False),
    ("MAGAIL-C+KL", True,  False, True),
]
SEEDS = [0, 1, 2, 3, 4]
PROGRESS_FILE = "ablation_progress.json"

# measured on this machine: ~21s/iteration training, ~17.5s/episode eval
SEC_PER_ITER = 21.0
SEC_PER_EVAL_EPISODE = 17.5


def estimate_run_seconds(n_iters, eval_every, eval_episodes):
    n_evals = n_iters // eval_every
    return n_iters * SEC_PER_ITER + n_evals * eval_episodes * SEC_PER_EVAL_EPISODE


def solve_iterations(budget_sec, n_runs, eval_every, eval_episodes, safety=0.85):
    """Largest n_iters fitting the budget, with a safety margin so a
    slightly-slow run doesn't overrun the hard restart deadline."""
    per_run = budget_sec * safety / n_runs
    lo, hi = 10, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_run_seconds(mid, eval_every, eval_episodes) <= per_run:
            lo = mid
        else:
            hi = mid - 1
    return lo


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=9.0)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override the budget-derived iteration count")
    args = ap.parse_args()

    plan = [(c, s, uc, sc, kl) for (c, uc, sc, kl) in CONDITIONS for s in SEEDS]
    # interleave by seed so a partial run still has BOTH sides of the
    # C-vs-SC comparison at every completed seed
    plan.sort(key=lambda x: (x[1], [c[0] for c in CONDITIONS].index(x[0])))

    budget_sec = args.budget_hours * 3600
    n_iters = args.iterations or solve_iterations(
        budget_sec, len(plan), args.eval_every, args.eval_episodes)
    per_run = estimate_run_seconds(n_iters, args.eval_every, args.eval_episodes)

    print(f"Budget: {args.budget_hours:.1f}h  |  runs: {len(plan)}  |  "
          f"iterations/run: {n_iters}")
    print(f"Estimated per run: {per_run/60:.1f} min  |  total: "
          f"{per_run*len(plan)/3600:.2f}h (85% safety margin applied)\n")
    for i, (c, s, uc, sc, kl) in enumerate(plan):
        print(f"  [{i+1:2d}] {c:14s} seed={s}  ctx={uc} shuffle={sc} kl={kl}")

    if args.dry_run:
        print("\nDRY RUN -- nothing launched.")
        return

    if n_iters < 50:
        print(f"\nWARNING: {n_iters} iterations/run is very short. The convergence "
              f"probe showed no SAP movement in 160 iterations WITH the style-gradient "
              f"bug present; with it fixed, behaviour may move faster, but this is "
              f"untested. Consider reducing scope (fewer conditions) rather than "
              f"running all 15 too briefly to show anything.")

    import finetune_loop
    prog = load_progress()
    t_start = time.time()

    for i, (cond, seed, use_ctx, shuf, kl) in enumerate(plan):
        key = f"{cond}_seed{seed}"
        if key in prog["completed"]:
            print(f"\n[{i+1}/{len(plan)}] {key} already complete -- skipping")
            continue

        elapsed = time.time() - t_start
        remaining_budget = budget_sec - elapsed
        runs_left = len(plan) - i
        print(f"\n[{i+1}/{len(plan)}] {key}  |  elapsed {elapsed/3600:.2f}h  |  "
              f"budget left {remaining_budget/3600:.2f}h  |  {runs_left} runs left")

        if remaining_budget < per_run * 0.5:
            print(f"STOPPING: insufficient budget remaining for a meaningful run. "
                  f"{runs_left} runs not started. Resume with the same command -- "
                  f"completed runs are skipped automatically.")
            break

        result = finetune_loop.run(
            condition=cond, seed=seed, max_iterations=n_iters,
            use_kl=kl, shuffle_context=shuf, use_context=use_ctx,
            eval_every=args.eval_every, eval_episodes=args.eval_episodes)

        prog["completed"].append(key)
        prog["results"].append(result)
        save_progress(prog)

    print(f"\nTotal elapsed: {(time.time()-t_start)/3600:.2f}h")
    print(f"Completed: {len(prog['completed'])}/{len(plan)} runs")
    print(f"Progress saved to {PROGRESS_FILE} -- rerun the same command to resume.")


if __name__ == '__main__':
    main()
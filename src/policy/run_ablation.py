"""
run_ablation.py

The locked 5-condition ablation matrix. Wraps finetune_loop.py's run()
logic per condition/seed rather than duplicating the training loop --
imports and calls it with condition-specific config.

MAX_ITERATIONS is READ FROM finetune_loop.py, currently a placeholder
(500). DO NOT run the real matrix until that number is replaced with
something justified by the convergence probe -- running 25+ ablation
jobs at an arbitrary iteration count wastes exactly the compute budget
this whole throughput-measurement exercise existed to protect.

Priority order (already decided, carried forward): MAGAIL-C vs MAGAIL-SC
first (the causal claim, the actual novelty), baseline second, MAGAIL-NC
third, lambda sweep last/first-to-cut if time runs short -- the sweep is
already only a "trend estimate" at 3 seeds by the methodology's own
admission.

Locked protocol: 5 seeds per headline condition, 3 seeds for the lambda
sweep (lambda in {0.001, 0.01, 0.1, 1.0}). DO NOT increase before Sep 16
-- pre-registering CSI_SAP as the sole primary endpoint is what makes
Holm-corrected significance achievable at n=5; extra seeds were
explicitly deferred to the Oct 1 AAMAS extension.
"""

import copy

CONDITIONS = [
    # (name, use_context, shuffle_context, use_kl, priority)
    ("MAGAIL-C",     True,  False, False, 1),   # true context, no KL yet -- the headline contrast
    ("MAGAIL-SC",    True,  True,  False, 1),   # SAME architecture, shuffled context -- causal control
    ("MAPPO-baseline", False, False, False, 2), # frozen, no fine-tuning at all
    ("MAGAIL-NC",    False, False, False, 3),   # fine-tuned WITHOUT context in the discriminator
    ("MAGAIL-C+KL",  True,  False, True,  4),   # true context + KL anchor, lambda from the sweep below
]

LAMBDA_SWEEP = [0.001, 0.01, 0.1, 1.0]
SEEDS_HEADLINE = [0, 1, 2, 3, 4]
SEEDS_SWEEP = [0, 1, 2]


def build_run_plan():
    """Returns an ORDERED list of (condition_name, seed, config_overrides)
    -- ordered by priority first (so a time-constrained run naturally does
    the highest-value work first), matching the already-decided triage."""
    plan = []

    # MAGAIL-C and MAGAIL-SC first, interleaved by seed so a partial run
    # has BOTH sides of the causal comparison at each seed, not all of one
    # then none of the other
    for seed in SEEDS_HEADLINE:
        plan.append(("MAGAIL-C", seed, {"use_context": True, "shuffle_context": False, "use_kl": False}))
        plan.append(("MAGAIL-SC", seed, {"use_context": True, "shuffle_context": True, "use_kl": False}))

    for seed in SEEDS_HEADLINE:
        plan.append(("MAPPO-baseline", seed, {"use_context": False, "shuffle_context": False, "use_kl": False, "frozen": True}))

    for seed in SEEDS_HEADLINE:
        plan.append(("MAGAIL-NC", seed, {"use_context": False, "shuffle_context": False, "use_kl": False}))

    # KL condition needs a lambda -- selected from the sweep's own result,
    # per the methodology's own ordering (sweep informs lambda choice for
    # this condition). Left as a TODO: plug in the actual selected lambda
    # once the sweep below has run and been read.
    for seed in SEEDS_HEADLINE:
        plan.append(("MAGAIL-C+KL", seed, {"use_context": True, "shuffle_context": False, "use_kl": True, "lam": None}))  # TODO: lam from sweep

    # sweep LAST -- lowest priority, cut first if time runs short, only 3
    # seeds by design (a trend estimate, not confirmatory)
    for lam in LAMBDA_SWEEP:
        for seed in SEEDS_SWEEP:
            plan.append((f"lambda_sweep_{lam}", seed, {"use_context": True, "shuffle_context": False, "use_kl": True, "lam": lam}))

    return plan


def run_all(dry_run=True):
    """dry_run=True: just print the plan and total run count, don't
    actually launch anything -- use this FIRST to sanity-check the plan
    and multiply by the per-run duration from the convergence probe
    before committing real compute."""
    plan = build_run_plan()
    n_headline = len(CONDITIONS) * len(SEEDS_HEADLINE)
    n_sweep = len(LAMBDA_SWEEP) * len(SEEDS_SWEEP)
    print(f"Total runs planned: {len(plan)}")
    print(f"  headline conditions ({len(CONDITIONS)} conditions x {len(SEEDS_HEADLINE)} seeds): {n_headline} runs")
    print(f"  lambda sweep ({len(LAMBDA_SWEEP)} values x {len(SEEDS_SWEEP)} seeds): {n_sweep} runs")
    print(f"  TOTAL: {n_headline + n_sweep} runs")
    assert n_headline + n_sweep == len(plan), "summary count and actual plan length disagree -- fix before trusting either"
    print()
    for i, (name, seed, cfg) in enumerate(plan):
        print(f"  [{i+1:2d}] {name:20s} seed={seed}  {cfg}")

    if dry_run:
        print("\nDRY RUN -- nothing launched. Call run_all(dry_run=False) once")
        print("MAX_ITERATIONS in finetune_loop.py is a real, justified number,")
        print("not the current 500-iteration placeholder.")
        return plan

    import finetune_loop
    for name, seed, cfg in plan:
        print(f"\n{'='*70}\nSTARTING: {name}, seed={seed}\n{'='*70}")
        # NOTE: finetune_loop.run() currently takes no arguments -- this
        # call signature is a TODO once finetune_loop.py is parameterized
        # to accept (condition_config, seed, run_name) rather than reading
        # its module-level constants directly. Flagged, not silently
        # assumed to already work.
        raise NotImplementedError(
            "finetune_loop.run() needs to accept condition_config/seed/run_name "
            "parameters before this can actually launch runs -- currently reads "
            "fixed module-level constants (MAGAIL-C-equivalent config, no seeding). "
            "Parameterize finetune_loop.py first."
        )


if __name__ == '__main__':
    run_all(dry_run=True)
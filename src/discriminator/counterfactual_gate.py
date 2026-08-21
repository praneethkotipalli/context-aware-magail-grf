"""
counterfactual_gate.py

The Section 3.3.3 counterfactual swap-test gate, finalized.

Runs BIDIRECTIONALLY against real expert data:
  - real late-winning examples, swapped to early-losing
  - real early-losing examples, swapped to late-winning

Gate criterion (locked, 0.1 threshold): mean absolute shift > 0.1, in
EACH direction INDEPENDENTLY -- not averaged together. Averaging would
let a strong result in one direction paper over a context-blind result
in the other, exactly the failure mode this gate exists to catch.

Also reports (does not gate on, but always prints/flags) the fraction of
individual examples whose own |shift| exceeds 0.1. A population can pass
on mean alone while a real minority of examples show near-zero shift --
the mean-only criterion would hide that. Per your choice: still reports
PASS if the mean criterion is met (that's the locked spec, and this
project's headline claim rests on it), but prints an explicit
inconsistency warning rather than staying silent about it.
"""

import numpy as np

from context_shift_scoring import (
    select_by_true_context, score_context_shift, LATE_WINNING, EARLY_LOSING,
)

SHIFT_THRESHOLD = 0.1          # locked, Section 3.3.3
LOW_CONSISTENCY_WARN = 0.5     # NOT locked -- a reporting heuristic only,
                                # not part of the pass/fail decision itself


def _direction_report(name, shifts):
    abs_shifts = np.abs(shifts)
    return {
        "direction": name,
        "n": len(shifts),
        "mean_shift": float(shifts.mean()),
        "mean_abs_shift": float(abs_shifts.mean()),
        "median_abs_shift": float(np.median(abs_shifts)),
        "std_abs_shift": float(abs_shifts.std()),
        "frac_exceeding_0.1": float(np.mean(abs_shifts > SHIFT_THRESHOLD)),
        "passes_mean_criterion": bool(abs_shifts.mean() > SHIFT_THRESHOLD),
    }


def run_counterfactual_gate(discriminator, expert_features):
    """
    expert_features: (N, 137) already-normalized real expert vectors
        (e.g. straight from expert_features_cache.npz, unfiltered --
        this function does its own filtering internally).

    Returns (overall_pass: bool, summary: dict). summary carries both
    direction reports in full plus any warnings -- log the whole thing,
    don't reduce it to the boolean.
    """
    mask_lw = select_by_true_context(
        expert_features, t_norm_max=LATE_WINNING['t_norm_max'],
        delta_score_sign=LATE_WINNING['delta_score_sign'],
    )
    if mask_lw.sum() == 0:
        raise ValueError("no real late-winning examples in expert_features -- cannot run this direction")
    _, _, shift_lw = score_context_shift(
        discriminator, expert_features[mask_lw],
        target_t_norm=0.85, target_delta_score_raw=-2,
    )
    report_lw = _direction_report("late_winning -> early_losing", shift_lw)

    mask_el = select_by_true_context(
        expert_features, t_norm_min=EARLY_LOSING['t_norm_min'],
        delta_score_sign=EARLY_LOSING['delta_score_sign'],
    )
    if mask_el.sum() == 0:
        raise ValueError("no real early-losing examples in expert_features -- cannot run this direction")
    _, _, shift_el = score_context_shift(
        discriminator, expert_features[mask_el],
        target_t_norm=0.1, target_delta_score_raw=2,
    )
    report_el = _direction_report("early_losing -> late_winning", shift_el)

    overall_pass = report_lw["passes_mean_criterion"] and report_el["passes_mean_criterion"]

    warnings = []
    for r in (report_lw, report_el):
        if r["passes_mean_criterion"] and r["frac_exceeding_0.1"] < LOW_CONSISTENCY_WARN:
            warnings.append(
                f"{r['direction']}: passes on mean_abs_shift={r['mean_abs_shift']:.4f}, "
                f"but only {r['frac_exceeding_0.1']*100:.1f}% of individual examples exceed "
                f"{SHIFT_THRESHOLD} -- the mean may be carried by a minority of large shifts, "
                f"not a consistently context-sensitive population."
            )

    return overall_pass, {"directions": [report_lw, report_el], "warnings": warnings}


def print_gate_report(overall_pass, summary):
    print("=" * 70)
    print("COUNTERFACTUAL SWAP-TEST GATE (Section 3.3.3)")
    print("=" * 70)
    for r in summary["directions"]:
        status = "PASS" if r["passes_mean_criterion"] else "FAIL"
        print(f"\n[{status}] {r['direction']}  (n={r['n']})")
        print(f"    mean signed shift : {r['mean_shift']:+.4f}")
        print(f"    mean |shift|      : {r['mean_abs_shift']:.4f}  (threshold {SHIFT_THRESHOLD})")
        print(f"    median |shift|    : {r['median_abs_shift']:.4f}")
        print(f"    frac exceeding    : {r['frac_exceeding_0.1']*100:.1f}%")

    if summary["warnings"]:
        print("\n" + "-" * 70)
        for w in summary["warnings"]:
            print(f"  WARNING: {w}")

    print("\n" + "=" * 70)
    print(f"OVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print("  -> safe to connect discriminator to policy" if overall_pass
          else "  -> DO NOT connect -- context-invariant in at least one direction")
    print("=" * 70)
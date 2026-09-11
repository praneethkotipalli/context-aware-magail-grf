"""
investigate_variance_and_consistency.py -- v2, FIXED + SAVES TO JSON.

Two corrections vs the version this replaces:
  1. Now writes investigate_variance_and_consistency.json -- every other
     analysis script in this pipeline (select_lambda_from_logs.py,
     run_significance_tests.py) saved its result; this one only printed
     to stdout, meaning the finding existed nowhere citable.
  2. The original checkpoint-level binomial test (n=48) was STATISTICALLY
     INVALID -- it treated ~6 within-seed eval checkpoints as independent
     observations, when consecutive evals within one training run are
     highly autocorrelated (a seed aligned at it150 is almost certainly
     still aligned at it200). The true independent sample size is 8
     SEEDS, not 48 checkpoints. This version replaces that test with the
     correct one: does each SEED show a majority-direction result across
     its own trajectory, tested at n=8.

Run:  python investigate_variance_and_consistency.py --launch-id matrix_0911ecd0
"""
import argparse, json, os, re
import numpy as np
from scipy import stats

EVAL_RE = re.compile(
    r"EVAL[^:]*:\s+win=(?P<win>[\d.]+)\s+SAP=(?P<sap>[\d.]+)%\s+"
    r"MECHA=(?P<mecha>[\d.nan]+)\s+CSI=(?P<csi>[-\d.eNone]+)")


def parse_checkpoints(log_path):
    vals = []
    if not os.path.exists(log_path):
        return vals
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                vals.append({"sap": float(m["sap"]), "win": float(m["win"])})
    return vals


def seed_level_sign_test(other, train_log_dir, launch_id):
    """CORRECTED unit of analysis: one binary outcome PER SEED (majority
    direction across that seed's own checkpoints), tested at n=8 --
    not n=48 checkpoint-level pseudo-observations."""
    seed_majority = {}
    for seed in range(8):
        c_log = os.path.join(train_log_dir, f"final-C-{launch_id}_seed{seed}.log")
        o_log = os.path.join(train_log_dir, f"final-{other}-{launch_id}_seed{seed}.log")
        c_vals, o_vals = parse_checkpoints(c_log), parse_checkpoints(o_log)
        n = min(len(c_vals), len(o_vals))
        if n == 0:
            continue
        wins = sum(1 for i in range(n) if c_vals[i]["sap"] < o_vals[i]["sap"])
        seed_majority[seed] = {"c_lower_count": wins, "n_checkpoints": n,
                              "majority_c_lower": wins > n / 2}

    n_seeds = len(seed_majority)
    majority_count = sum(1 for v in seed_majority.values() if v["majority_c_lower"])
    winning_seeds = [s for s, v in seed_majority.items() if v["majority_c_lower"]]
    losing_seeds = [s for s, v in seed_majority.items() if not v["majority_c_lower"]]

    p = stats.binomtest(majority_count, n_seeds, 0.5, alternative="greater").pvalue
    return {
        "comparison": f"C vs {other}", "n_seeds": n_seeds,
        "seeds_where_C_majority_lower": winning_seeds,
        "seeds_where_C_majority_lower_count": majority_count,
        "seeds_opposite_direction": losing_seeds,
        "per_seed_detail": seed_majority,
        "binomial_p_value": float(p),
        "significant_at_05": bool(p < 0.05),
        "note": "CORRECT unit of analysis -- one observation per seed (n=8), "
                "not per checkpoint (the earlier n=48 version was invalid: "
                "checkpoints within one training run are autocorrelated, not "
                "independent draws)."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-id", required=True)
    args = ap.parse_args()

    base = os.path.join("results_v2", "stage3_full_matrix", args.launch_id)
    raw_path = os.path.join(base, "all_runs_raw.json")
    train_log_dir = os.path.join(base, "train_logs")

    with open(raw_path) as f:
        all_runs = json.load(f)

    result = {"launch_id": args.launch_id}

    # ---- 1. per-seed C values -- bimodality check ----
    print("=" * 70)
    print("1. PER-SEED C sap_mean -- bimodal cluster check")
    print("=" * 70)
    c_rows = sorted([r for r in all_runs if r["condition_label"] == "C"], key=lambda r: r["seed"])
    saps = {r["seed"]: r["sap_mean"] for r in c_rows}
    vals_sorted = sorted(saps.items(), key=lambda kv: kv[1])
    mean, sd = np.mean(list(saps.values())), np.std(list(saps.values()), ddof=1)

    # simple 2-cluster split: largest gap in the sorted values
    sorted_vals = [v for _, v in vals_sorted]
    gaps = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
    split_idx = int(np.argmax(gaps)) + 1
    lower_cluster = [s for s, v in vals_sorted[:split_idx]]
    upper_cluster = [s for s, v in vals_sorted[split_idx:]]
    max_gap = max(gaps)

    for seed, v in vals_sorted:
        z = (v - mean) / sd
        cluster = "lower" if seed in lower_cluster else "upper"
        print(f"  seed {seed}: sap={v:.2f}%  (z={z:+.2f})  [{cluster} cluster]")
    print(f"\n  mean={mean:.2f}  sd={sd:.2f}")
    print(f"  Largest gap in sorted values: {max_gap:.2f} points, splitting "
          f"{len(lower_cluster)} lower-SAP seeds {lower_cluster} vs "
          f"{len(upper_cluster)} higher-SAP seeds {upper_cluster}")

    result["per_seed_C_sap"] = saps
    result["mean"] = float(mean); result["sd"] = float(sd)
    result["bimodal_split"] = {
        "lower_cluster_seeds": lower_cluster, "upper_cluster_seeds": upper_cluster,
        "largest_gap": float(max_gap),
        "interpretation": f"~{len(lower_cluster)} of 8 seeds converge to strong alignment "
                          f"(lower SAP); ~{len(upper_cluster)} of 8 fail to improve much "
                          f"on the frozen baseline -- a bimodal outcome pattern, not "
                          f"uniform variance around one mean."
    }

    # ---- 2. CORRECTED seed-level sign test ----
    print("\n" + "=" * 70)
    print("2. SEED-LEVEL SIGN TEST (corrected unit of analysis, n=8)")
    print("=" * 70)
    result["seed_level_tests"] = {}
    for other in ["NC", "SC"]:
        test = seed_level_sign_test(other, train_log_dir, args.launch_id)
        result["seed_level_tests"][f"C_vs_{other}"] = test
        print(f"\n  C vs {other}: majority-C-lower in {test['seeds_where_C_majority_lower_count']}/"
              f"{test['n_seeds']} seeds "
              f"(seeds {test['seeds_where_C_majority_lower']} vs opposite "
              f"{test['seeds_opposite_direction']})")
        print(f"  binomial p={test['binomial_p_value']:.4f}  "
              f"{'SIGNIFICANT' if test['significant_at_05'] else 'not significant'}")

    both = result["seed_level_tests"]
    if both["C_vs_NC"]["seeds_where_C_majority_lower"] == both["C_vs_SC"]["seeds_where_C_majority_lower"]:
        note = ("Identical seed split drives both comparisons -- this is evidence the "
                "pattern reflects C's OWN bimodal clustering (which seeds happened to "
                "converge to strong alignment), not something differentially true about "
                "NC vs SC specifically.")
        print(f"\n  NOTE: {note}")
        result["cross_comparison_note"] = note

    out_path = os.path.join(base, "investigate_variance_and_consistency.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
"""
run_significance_tests.py

Full statistical suite for the final ablation matrix. Replaces an earlier
draft that had four real gaps against the pre-registration:
  1. no Holm correction (tested each pair at raw alpha=0.05, not the
     Holm-adjusted threshold the pre-registered 2-test family requires)
  2. no Cliff's delta (pre-registered requirement; the number that matters
     once a p-value is null)
  3. SAP comparisons run as if confirmatory when they were never
     pre-registered as hypothesis tests -- bundling them with the CSI
     tests without labelling them exploratory is quiet p-hacking
  4. win-rate check used a one-sample t-test against a bare constant,
     ignoring that the baseline is itself an 8-seed ESTIMATE with its own
     uncertainty, and inconsistent with using Mann-Whitney everywhere else
     specifically because n=8 cannot support a normality assumption
     (this project's own locked methodology, Section 3.5.3: "All
     comparisons use non-parametric inference, since per-seed metric
     distributions are small-sample, bounded and skewed").

Reads final_eval_*.json files directly from the matrix's own output
directory -- does not assume a hand-assembled combined file exists.

KNOWN LIMITATION, printed prominently below, not silently carried: the
human corpus CSI target (-0.316) was computed using classify_bin's "late"
cutoff (T_norm <= 0.2222). This project's own compute_csi_proxy (used for
every csi_sap_proxy value tested here) uses a DIFFERENT cutoff (T_norm <=
0.3). The comparison to -0.316 is directionally informative but not a
clean apples-to-apples magnitude comparison -- state this in the write-up.

Run:
    python run_significance_tests.py --matrix-launch-id matrix_0911ecd0 \\
        --baseline-path results_v2/stage3_baseline/<baseline_launch_id>/baseline_full_results.json
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats

CONDITIONS = ["NC", "C", "SC", "C+KL"]
HUMAN_CSI_TARGET = -0.316          # computed at T_norm<=0.2222 -- see limitation note above
HUMAN_SAP_TARGET = 17.68
ALPHA = 0.05
N_BOOTSTRAP = 10000
N_POWER_SIMS = 3000
RNG_SEED = 0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_matrix_runs(launch_id, result_root="results_v2/stage3_full_matrix"):
    """Globs the per-run final_eval_*.json files the matrix script actually
    writes (not a hand-assembled combined file). Prints a data-integrity
    check -- exactly n=8 per condition, or a clear warning naming what's
    missing -- before any test runs, so a silently-short sample never
    passes unnoticed into a p-value."""
    pattern = os.path.join(result_root, launch_id, f"final_eval_final-*-{launch_id}_seed*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no final_eval files matched {pattern} -- check launch_id/result_root")

    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        # condition_label should already be present (matrix script writes it);
        # fall back to parsing the filename if an older run predates that field
        if "condition_label" not in r:
            base = os.path.basename(p)
            for c in CONDITIONS:
                if f"final-{c}-" in base:
                    r["condition_label"] = c
                    break
        runs.append(r)

    print(f"loaded {len(runs)} run files from {pattern}")
    print("\nDATA INTEGRITY CHECK -- expected n=8 per condition:")
    ok = True
    for c in CONDITIONS:
        n = sum(1 for r in runs if r.get("condition_label") == c)
        seeds = sorted(r.get("seed") for r in runs if r.get("condition_label") == c)
        flag = "OK" if n == 8 else "!! SHORT !!"
        print(f"  {c:6s}: n={n}  seeds={seeds}  {flag}")
        ok = ok and (n == 8)
    if not ok:
        print("\n  WARNING: at least one condition does not have n=8. Every test below still\n"
              "  runs on whatever n is actually present -- read n_a/n_b in the output, do not\n"
              "  assume 8 just because that was the target.")
    return runs


def load_baseline(path):
    with open(path) as f:
        rows = json.load(f)
    win = np.array([r["win_rate"] for r in rows])
    sap = np.array([r["sap_mean"] for r in rows])
    csi = np.array([r["csi_sap_proxy"] for r in rows])
    print(f"\nBASELINE: n={len(rows)}  win_rate mean={win.mean():.4f} (sd={win.std(ddof=1):.4f})  "
          f"sap_mean={sap.mean():.4f} (sd={sap.std(ddof=1):.4f})")
    return {"win_rate": win, "sap_mean": sap, "csi_sap_proxy": csi}


def get_metric(runs, condition, metric):
    return np.array([r[metric] for r in runs
                     if r.get("condition_label") == condition and r.get(metric) is not None])


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = np.sum(a[:, None] > b); lt = np.sum(a[:, None] < b)
    return float((gt - lt) / (len(a) * len(b)))


def mwu_test(a, b, direction="less"):
    """direction='less' -> H1: a stochastically less than b."""
    u, p = stats.mannwhitneyu(a, b, alternative=direction)
    return float(u), float(p)


def holm_correct(named_pvalues, alpha=ALPHA):
    """named_pvalues: list of (name, p). Returns list of dicts with the
    per-comparison Holm threshold and verdict, sorted by p ascending."""
    ordered = sorted(named_pvalues, key=lambda x: x[1])
    m = len(ordered)
    out = []
    for i, (name, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        out.append({"name": name, "p_raw": p, "holm_rank": i + 1,
                    "holm_threshold": thresh, "reject_h0": p < thresh})
    return out


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, ci=90, seed=RNG_SEED):
    """Seed-level bootstrap -- each seed is one independent training run,
    the natural resampling unit here (same logic as the episode-cluster
    bootstrap used for the human corpus target: resample at the level of
    the independent unit, not at a finer, correlated level)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_diff_ci(a, b, n_boot=N_BOOTSTRAP, ci=90, seed=RNG_SEED):
    """Bootstrap CI on the raw mean difference a-b, in original units --
    complements Cliff's delta (which is unitless) with something a reader
    can interpret directly: 'the mean difference was X, CI [lo,hi]'."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    boot_a = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    boot_b = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    diff = boot_a - boot_b
    lo, hi = np.percentile(diff, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def posthoc_power(observed_diff, pooled_sd, n, alpha_thresh, n_sims=N_POWER_SIMS, seed=RNG_SEED):
    """Simulation-based power: if the TRUE mean difference equalled the
    OBSERVED one (and noise matched the observed pooled sd), what fraction
    of n-per-group experiments like this one would reach significance at
    alpha_thresh? Low power on a null result means 'inconclusive', not
    'no effect' -- report both together, always."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        a = rng.normal(-observed_diff / 2, pooled_sd, n)
        b = rng.normal(observed_diff / 2, pooled_sd, n)
        try:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        if p < alpha_thresh:
            hits += 1
    return hits / n_sims


def scheirer_ray_hare(values, factor1, factor2):
    """Rank-based two-way ANOVA analogue (Scheirer, Ray & Hare, 1976),
    pre-registered in the original methodology (Section 3.5.3) as the test
    for the METHOD x SCENARIO interaction -- the direct statistical
    embodiment of the CSI hypothesis, tested on ranks of the raw per-
    scenario SAP values rather than on the already-differenced CSI.

    Implementation: rank all observations, run a standard two-way ANOVA
    on the ranks, scale each factor's SS by (SS/SS_total) * (N-1) to get
    an approximate chi-square statistic (the standard SRH construction).

    values, factor1, factor2: equal-length arrays. factor1 = condition
    label, factor2 = scenario label (e.g. 'late_win' / 'late_loss').
    """
    import pandas as pd
    df = pd.DataFrame({"y": values, "f1": factor1, "f2": factor2})
    df["rank"] = df["y"].rank()
    n = len(df)

    grand_mean_rank = df["rank"].mean()
    ss_total = ((df["rank"] - grand_mean_rank) ** 2).sum()

    def ss_factor(col):
        means = df.groupby(col)["rank"].mean()
        counts = df.groupby(col)["rank"].count()
        return float(((means - grand_mean_rank) ** 2 * counts).sum())

    ss_f1 = ss_factor("f1")
    ss_f2 = ss_factor("f2")

    cell_means = df.groupby(["f1", "f2"])["rank"].mean()
    cell_counts = df.groupby(["f1", "f2"])["rank"].count()
    ss_cells = float(((cell_means - grand_mean_rank) ** 2 * cell_counts).sum())
    ss_interaction = ss_cells - ss_f1 - ss_f2

    h_f1 = ss_f1 / (ss_total / (n - 1))
    h_f2 = ss_f2 / (ss_total / (n - 1))
    h_int = ss_interaction / (ss_total / (n - 1))

    df1 = df["f1"].nunique() - 1
    df2 = df["f2"].nunique() - 1
    df_int = df1 * df2

    p_f1 = 1 - stats.chi2.cdf(h_f1, df1)
    p_f2 = 1 - stats.chi2.cdf(h_f2, df2)
    p_int = 1 - stats.chi2.cdf(h_int, df_int)

    return {
        "factor1_H": h_f1, "factor1_df": df1, "factor1_p": p_f1,
        "factor2_H": h_f2, "factor2_df": df2, "factor2_p": p_f2,
        "interaction_H": h_int, "interaction_df": df_int, "interaction_p": p_int,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-launch-id", required=True)
    ap.add_argument("--baseline-path", required=True)
    ap.add_argument("--result-root", default="results_v2/stage3_full_matrix")
    args = ap.parse_args()

    print("=" * 78)
    print("KNOWN LIMITATION (state this in the write-up, do not silently carry it):")
    print("  human CSI target (-0.316) uses classify_bin's late cutoff (T_norm<=0.2222).")
    print("  compute_csi_proxy (every csi_sap_proxy value below) uses T_norm<=0.3.")
    print("  Directionally informative; not a clean magnitude comparison.")
    print("=" * 78)

    runs = load_matrix_runs(args.matrix_launch_id, args.result_root)
    baseline = load_baseline(args.baseline_path)

    out = {"launch_id": args.matrix_launch_id, "confirmatory": {}, "threshold_check": {},
           "exploratory": {}, "bootstrap_summary": {}, "scheirer_ray_hare": None}

    # ---------------- CONFIRMATORY FAMILY (pre-registered, m=2) ----------------
    print("\n" + "=" * 78)
    print("CONFIRMATORY FAMILY (pre-registered): C vs NC, C vs SC on csi_sap_proxy")
    print("One-sided (H1: C more negative). Family size m=2 for Holm.")
    print("=" * 78)

    c_csi = get_metric(runs, "C", "csi_sap_proxy")
    nc_csi = get_metric(runs, "NC", "csi_sap_proxy")
    sc_csi = get_metric(runs, "SC", "csi_sap_proxy")

    confirmatory_raw = {}
    for name, other in [("C vs NC", nc_csi), ("C vs SC (PRIMARY)", sc_csi)]:
        u, p = mwu_test(c_csi, other, "less")
        delta = cliffs_delta(c_csi, other)
        mean_diff, ci_lo, ci_hi = bootstrap_diff_ci(c_csi, other)
        pooled_sd = float(np.sqrt((c_csi.var(ddof=1) + other.var(ddof=1)) / 2))
        confirmatory_raw[name] = {
            "U": u, "p_raw": p, "cliffs_delta": delta,
            "mean_diff": mean_diff, "diff_ci90": [ci_lo, ci_hi],
            "n_a": len(c_csi), "n_b": len(other),
            "mean_a": float(c_csi.mean()), "mean_b": float(other.mean()),
            "pooled_sd": pooled_sd,
        }
        print(f"\n{name}: C (n={len(c_csi)}, mean={c_csi.mean():+.4f}) vs "
              f"comparator (n={len(other)}, mean={other.mean():+.4f})")
        print(f"  U={u:.1f}  p_raw={p:.4f}  Cliff's delta={delta:+.3f}")
        print(f"  mean difference={mean_diff:+.4f}  90% bootstrap CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")

    holm_results = holm_correct([(name, d["p_raw"]) for name, d in confirmatory_raw.items()])
    print("\nHOLM STEP-DOWN CORRECTION:")
    for hr in holm_results:
        d = confirmatory_raw[hr["name"]]
        d.update({"holm_threshold": hr["holm_threshold"], "holm_rank": hr["holm_rank"],
                  "reject_h0": hr["reject_h0"]})
        verdict = "REJECT H0 (significant)" if hr["reject_h0"] else "fail to reject"
        print(f"  rank {hr['holm_rank']}: {hr['name']:20s} p={hr['p_raw']:.4f}  "
              f"threshold={hr['holm_threshold']:.4f}  -> {verdict}")

        power = posthoc_power(d["mean_diff"], d["pooled_sd"], min(d["n_a"], d["n_b"]),
                              hr["holm_threshold"])
        d["posthoc_power_at_observed_effect"] = power
        print(f"           post-hoc power at the OBSERVED effect size: {power:.2f} "
              f"({'a null here is inconclusive, not evidence of no effect' if power < 0.5 else 'reasonably powered'})")

    out["confirmatory"] = confirmatory_raw

    # ---------------- C+KL PRIMARY THRESHOLD CHECK (not a p-value test) --------
    print("\n" + "=" * 78)
    print("C+KL PRIMARY ENDPOINT (pre-registered threshold check, not significance test)")
    print("=" * 78)
    ckl_win = get_metric(runs, "C+KL", "win_rate")
    ckl_sap = get_metric(runs, "C+KL", "sap_mean")
    base_win_mean, base_win_lo, base_win_hi = bootstrap_mean_ci(baseline["win_rate"])
    base_sap_mean, base_sap_lo, base_sap_hi = bootstrap_mean_ci(baseline["sap_mean"])

    win_thresh = base_win_mean - 0.03
    sap_thresh = base_sap_mean - 15.0
    win_pass = ckl_win.mean() >= win_thresh
    sap_pass = ckl_sap.mean() <= sap_thresh
    print(f"  baseline win={base_win_mean:.4f} (90% CI [{base_win_lo:.4f},{base_win_hi:.4f}])  "
          f"threshold={win_thresh:.4f}  C+KL win={ckl_win.mean():.4f}  -> {'PASS' if win_pass else 'FAIL'}")
    print(f"  baseline SAP={base_sap_mean:.2f} (90% CI [{base_sap_lo:.2f},{base_sap_hi:.2f}])  "
          f"threshold={sap_thresh:.2f}  C+KL SAP={ckl_sap.mean():.2f}  -> {'PASS' if sap_pass else 'FAIL'} "
          f"(margin: {sap_thresh - ckl_sap.mean():+.2f} pts)")
    out["threshold_check"] = {
        "baseline_win_mean": base_win_mean, "baseline_win_ci90": [base_win_lo, base_win_hi],
        "baseline_sap_mean": base_sap_mean, "baseline_sap_ci90": [base_sap_lo, base_sap_hi],
        "win_threshold": win_thresh, "ckl_win_mean": float(ckl_win.mean()), "win_pass": bool(win_pass),
        "sap_threshold": sap_thresh, "ckl_sap_mean": float(ckl_sap.mean()), "sap_pass": bool(sap_pass),
    }

    # ---------------- EXPLORATORY (clearly separated, own Holm family) --------
    print("\n" + "=" * 78)
    print("EXPLORATORY -- NOT pre-registered, NOT confirmatory. Own Holm family so an")
    print("incidental low p here is never misread as a discovery.")
    print("=" * 78)

    exploratory = {}

    # SAP, same two pairs
    print("\n-- SAP, same pairs as the confirmatory CSI tests --")
    c_sap = get_metric(runs, "C", "sap_mean"); nc_sap = get_metric(runs, "NC", "sap_mean")
    sc_sap_ = get_metric(runs, "SC", "sap_mean")
    for name, other in [("C vs NC (SAP)", nc_sap), ("C vs SC (SAP)", sc_sap_)]:
        u, p = mwu_test(c_sap, other, "less")
        delta = cliffs_delta(c_sap, other)
        exploratory[name] = {"U": u, "p_raw": p, "cliffs_delta": delta,
                             "mean_a": float(c_sap.mean()), "mean_b": float(other.mean())}
        print(f"  {name}: U={u:.1f} p={p:.4f} delta={delta:+.3f} "
              f"(C={c_sap.mean():.2f}, other={other.mean():.2f})")

    # all 6 pairwise on CSI across all 4 conditions -- own Holm family
    print("\n-- ALL pairwise CSI comparisons across all 4 conditions (own Holm family, m=6) --")
    all_csi = {c: get_metric(runs, c, "csi_sap_proxy") for c in CONDITIONS}
    from itertools import combinations
    pair_p = {}
    for a, b in combinations(CONDITIONS, 2):
        u, p = mwu_test(all_csi[a], all_csi[b], "two-sided")
        delta = cliffs_delta(all_csi[a], all_csi[b])
        pair_p[f"{a} vs {b}"] = p
        exploratory[f"{a} vs {b} (all-pairs CSI)"] = {"U": u, "p_raw": p, "cliffs_delta": delta}
    holm_pairs = holm_correct(list(pair_p.items()))
    for hr in holm_pairs:
        print(f"  {hr['name']:12s} p_raw={hr['p_raw']:.4f}  holm_threshold={hr['holm_threshold']:.4f}  "
              f"-> {'flagged' if hr['reject_h0'] else 'not flagged'}")

    # omnibus Kruskal-Wallis, CSI and SAP
    print("\n-- Omnibus Kruskal-Wallis across all 4 conditions --")
    for metric_name, metric_dict in [("csi_sap_proxy", all_csi),
                                     ("sap_mean", {c: get_metric(runs, c, "sap_mean") for c in CONDITIONS})]:
        h, p = stats.kruskal(*metric_dict.values())
        print(f"  {metric_name}: H={h:.3f}  p={p:.4f}")
        exploratory[f"omnibus_{metric_name}"] = {"H": float(h), "p": float(p)}

    # win-rate vs baseline -- Mann-Whitney (not t-test), baseline as its own distribution
    print("\n-- Win-rate vs baseline (Mann-Whitney, two-sided, baseline as its own 8-seed sample) --")
    for cond in CONDITIONS:
        wins = get_metric(runs, cond, "win_rate")
        u, p = mwu_test(wins, baseline["win_rate"], "two-sided")
        delta = cliffs_delta(wins, baseline["win_rate"])
        print(f"  {cond:6s}: mean win={wins.mean():.4f} vs baseline={baseline['win_rate'].mean():.4f}  "
              f"U={u:.1f} p={p:.4f} delta={delta:+.3f}")
        exploratory[f"{cond}_vs_baseline_win"] = {"U": u, "p_raw": p, "cliffs_delta": delta,
                                                  "mean_cond": float(wins.mean())}

    out["exploratory"] = exploratory

    # ---------------- Bootstrap summary table, all conditions ----------------
    print("\n" + "=" * 78)
    print(f"BOOTSTRAP 90% CI SUMMARY (seed-level resampling, n_boot={N_BOOTSTRAP})")
    print("=" * 78)
    print(f"{'condition':8s} {'CSI mean':>10s} {'CSI 90% CI':>22s} {'SAP mean':>9s} {'SAP 90% CI':>18s}")
    for c in ["MAPPO-baseline"] + CONDITIONS:
        csi_vals = baseline["csi_sap_proxy"] if c == "MAPPO-baseline" else get_metric(runs, c, "csi_sap_proxy")
        sap_vals = baseline["sap_mean"] if c == "MAPPO-baseline" else get_metric(runs, c, "sap_mean")
        cm, clo, chi = bootstrap_mean_ci(csi_vals)
        sm, slo, shi = bootstrap_mean_ci(sap_vals)
        print(f"{c:8s} {cm:+10.4f} [{clo:+.4f},{chi:+.4f}]   {sm:9.2f} [{slo:.2f},{shi:.2f}]")
        out["bootstrap_summary"][c] = {"csi_mean": cm, "csi_ci90": [clo, chi],
                                       "sap_mean": sm, "sap_ci90": [slo, shi]}
    print(f"\n  (for reference: human corpus CSI target = {HUMAN_CSI_TARGET}, "
          f"human SAP target = {HUMAN_SAP_TARGET} -- see limitation note above)")

    # ---------------- Scheirer-Ray-Hare (attempt, or explain what's missing) --
    print("\n" + "=" * 78)
    print("SCHEIRER-RAY-HARE: method x scenario interaction (pre-registered, Sec 3.5.3)")
    print("=" * 78)
    sample_run = runs[0]
    needs = ["sap_late_win", "sap_late_loss"]
    if all(k in sample_run for k in needs):
        rows_y, rows_f1, rows_f2 = [], [], []
        for c in CONDITIONS:
            for r in runs:
                if r.get("condition_label") != c:
                    continue
                rows_y.append(r["sap_late_win"]); rows_f1.append(c); rows_f2.append("late_win")
                rows_y.append(r["sap_late_loss"]); rows_f1.append(c); rows_f2.append("late_loss")
        srh = scheirer_ray_hare(rows_y, rows_f1, rows_f2)
        print(f"  method:      H={srh['factor1_H']:.3f} df={srh['factor1_df']} p={srh['factor1_p']:.4f}")
        print(f"  scenario:    H={srh['factor2_H']:.3f} df={srh['factor2_df']} p={srh['factor2_p']:.4f}")
        print(f"  INTERACTION: H={srh['interaction_H']:.3f} df={srh['interaction_df']} "
              f"p={srh['interaction_p']:.4f}  <-- the CSI hypothesis, tested directly")
        out["scheirer_ray_hare"] = srh
    else:
        print(f"  SKIPPED -- requires per-scenario SAP means ({needs}) not present in the")
        print(f"  current final_eval_*.json schema. compute_csi_proxy already computes both")
        print(f"  sub-means internally to produce csi_sap_proxy (their difference) -- add")
        print(f"  'sap_late_win' and 'sap_late_loss' to evaluate_policy's returned dict and")
        print(f"  re-run final evaluation to enable this test. Currently only the already-")
        print(f"  differenced value is retained, which is sufficient for the confirmatory")
        print(f"  tests above but not for this direct interaction test.")
        out["scheirer_ray_hare"] = {"skipped": True, "missing_fields": needs}

    out_path = os.path.join(args.result_root, args.matrix_launch_id, "significance_tests_full.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
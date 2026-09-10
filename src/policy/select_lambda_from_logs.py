"""
select_lambda_from_logs.py

Fixes the collect_results() bug in run_lambda_sweep.py: result_*.json
only ever contained run metadata (condition, seed, wall_time_sec, ...),
never win_rate/sap_mean/eval_history -- so the JSON-preferring path
silently produced empty data. This script reads the .log files directly,
where the real "EVAL (...): win=... SAP=...%" lines live.

Run:  python select_lambda_from_logs.py --log-dir results_v2/stage2_lambda_sweep/<LAUNCH_ID>/logs
"""
import argparse, glob, json, os, re
import statistics as st

EVAL_RE = re.compile(
    r"EVAL[^:]*:\s+win=(?P<win>[\d.]+)\s+SAP=(?P<sap>[\d.]+)%\s+"
    r"MECHA=(?P<mecha>[\d.nan]+)\s+CSI=(?P<csi>[-\d.eNone]+)")
NAME_RE = re.compile(r"lsweep-lam(?P<lam>[\d.]+)-\S+?_seed(?P<seed>\d+)\.log$")

SAP_BASELINE = 85.9
SAP_MOVEMENT_FLOOR = 5.0
BASELINE_WIN_RATE = 0.624


def parse_log(path):
    evals = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", default="lambda_star_from_logs.json")
    args = ap.parse_args()

    by_lam = {}
    files = sorted(glob.glob(os.path.join(args.log_dir, "lsweep-lam*.log")))
    if not files:
        print(f"No lsweep-lam*.log files found in {args.log_dir} -- check the path.")
        return
    print(f"found {len(files)} log files\n")

    for path in files:
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            print(f"  SKIP (name doesn't match pattern): {path}")
            continue
        lam, seed = float(m["lam"]), int(m["seed"])
        evals = parse_log(path)
        if not evals:
            print(f"  WARNING: no EVAL lines found in {path} (job may have crashed "
                  f"before its first eval, or the eval_every/eval_episodes settings "
                  f"mean this run never hit an eval boundary)")
            continue
        final = evals[-1]
        by_lam.setdefault(lam, {"win": [], "sap": [], "n_evals": []})
        by_lam[lam]["win"].append(final["win_rate"])
        by_lam[lam]["sap"].append(final["sap_mean"])
        by_lam[lam]["n_evals"].append(len(evals))
        print(f"  lam={lam:<6g} seed={seed}  final: win={final['win_rate']:.3f} "
              f"SAP={final['sap_mean']:.2f}%  ({len(evals)} evals in log)")

    print(f"\n{'lambda':>8s} {'n':>2s} {'mean win':>10s} {'win SE':>8s} "
          f"{'mean SAP':>9s} {'moved?':>7s} {'eligible?':>10s}")
    candidates = []
    table = []
    for lam in sorted(by_lam):
        wins, saps = by_lam[lam]["win"], by_lam[lam]["sap"]
        mean_win = st.mean(wins)
        se_win = (st.stdev(wins) / len(wins) ** 0.5) if len(wins) > 1 else float("nan")
        mean_sap = st.mean(saps)
        moved = (SAP_BASELINE - mean_sap) >= SAP_MOVEMENT_FLOOR
        eligible = mean_win >= (BASELINE_WIN_RATE - (se_win if se_win == se_win else 0))
        print(f"{lam:>8.3g} {len(wins):>2d} {mean_win:>10.3f} {se_win:>8.3f} "
              f"{mean_sap:>9.2f} {'yes' if moved else 'no':>7s} "
              f"{'yes' if eligible and moved else 'no':>10s}")
        table.append({"lambda": lam, "n": len(wins), "mean_win": mean_win, "se_win": se_win,
                      "mean_sap": mean_sap, "moved": moved, "eligible": eligible and moved})
        if eligible and moved:
            candidates.append(lam)

    lam_star = min(candidates) if candidates else None
    print(f"\nlambda* = {lam_star}" if lam_star is not None else
          "\nNo lambda both win-preserving AND showing real movement -- "
          "report the curve, do not force a pick.")

    with open(args.out, "w") as f:
        json.dump({"lambda_star": lam_star, "table": table}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
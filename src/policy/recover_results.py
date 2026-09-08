"""
recover_results.py

The per-run result_*.json files only captured run metadata -- the eval
metrics (win rate, SAP, MECHA, CSI) were never added to the returned dict.
Nothing is lost: everything is in W&B, and the EVAL lines in
ablation_logs/*.log carry the same numbers in plain text.

Two recovery routes:
  --source wandb  (preferred)  full per-iteration history, every logged key
  --source logs   (fallback)   parses ablation_logs/*.log, eval points only,
                               works with no network and no W&B access

Writes:
  recovered_metrics_full.csv      one row per logged step, all keys
  recovered_metrics_final.csv     one row per run, final eval values
  recovered_summary.json          per-condition aggregates (mean/sd over seeds)
"""

import argparse, glob, json, os, re
import numpy as np
import pandas as pd

WANDB_PROJECT = "praneethkotipalli-university-of-warwick/magail-c"

# [MAGAIL-C+KL_seed3 it525] EVAL (945s): win=0.640 SAP=84.72% MECHA=0.1176 CSI=0.0252 health=saturating lam=0.1000
EVAL_RE = re.compile(
    r"\[(?P<run>[\w\-\+\.]+_seed\d+)\s+it(?P<it>\d+)\]\s+EVAL\s+\((?P<evsec>\d+)s\):\s+"
    r"win=(?P<win>[\d\.]+)\s+SAP=(?P<sap>[\d\.]+)%\s+MECHA=(?P<mecha>[\d\.nan]+)\s+"
    r"CSI=(?P<csi>[-\d\.eNone]+)\s+health=(?P<health>\w+)\s+lam=(?P<lam>[\d\.]+)"
)


def from_wandb(run_names_file=None):
    import wandb
    api = wandb.Api()

    target_names = None
    if run_names_file:
        with open(run_names_file) as f:
            manifest = json.load(f)
        target_names = [r["run_name"] for r in manifest if r.get("returncode") == 0]
        print(f"Using {run_names_file} as ground truth: {len(target_names)} confirmed-successful run names")

    all_runs = list(api.runs(WANDB_PROJECT))
    print(f"W&B project has {len(all_runs)} runs total (across every launch attempt this week)")

    if target_names is None:
        print("\nWARNING: no --run-names-file given. Falling back to a max_iterations>=100")
        print("heuristic, which does NOT deduplicate repeated launches of the same run name.")
        print("If the same run was launched multiple times at the real iteration count,")
        print("this WILL silently average duplicates together. Strongly prefer:")
        print("    --run-names-file ablation_parallel_summary.json")
        candidates = [r for r in all_runs if r.config.get("max_iterations", 0) >= 100]
    else:
        by_name = {}
        for r in all_runs:
            by_name.setdefault(r.name, []).append(r)
        candidates, missing = [], []
        for name in target_names:
            matches = by_name.get(name, [])
            if not matches:
                missing.append(name)
                continue
            latest = sorted(matches, key=lambda r: r.created_at)[-1]
            if len(matches) > 1:
                print(f"  {name}: {len(matches)} W&B runs found, using most recent "
                      f"(created {latest.created_at})")
            candidates.append(latest)
        if missing:
            print(f"\nWARNING: {len(missing)} run names in the manifest have NO matching "
                  f"W&B run: {missing}")
        print(f"\nMatched {len(candidates)}/{len(target_names)} expected runs")

        # Cross-check: all 37 should have launched together (same subprocess batch),
        # so their created_at timestamps should cluster tightly. If wandb.init()
        # silently failed for one process in the FINAL attempt, "most recent match"
        # would silently fall back to an OLDER, failed attempt's run for that name --
        # this catches that case instead of trusting the count alone.
        if len(candidates) > 1:
            import datetime
            times = sorted((r.created_at, r.name) for r in candidates)
            times_dt = [t[0] if isinstance(t[0], datetime.datetime)
                       else datetime.datetime.fromisoformat(str(t[0]).replace("Z", "+00:00"))
                       for t in times]
            span = (times_dt[-1] - times_dt[0]).total_seconds()
            median_gap = sorted(
                (times_dt[i+1] - times_dt[i]).total_seconds() for i in range(len(times_dt)-1)
            )[len(times_dt)//2]
            print(f"\nTiming check: {len(candidates)} matched runs span "
                  f"{span/60:.1f} min from first to last launch (median gap between "
                  f"consecutive launches: {median_gap:.1f}s)")
            latest_time = times_dt[-1]
            stale = [(name, t) for (t, name) in zip(times_dt, [n for _, n in times])
                    if (latest_time - t).total_seconds() > 1800]
            if stale:
                print(f"\nWARNING -- {len(stale)} 'matched' run(s) are >30min older than the "
                      f"most recent one. These are almost certainly from an EARLIER, FAILED "
                      f"attempt (wandb.init likely failed for these in the real final launch):")
                for name, t in stale:
                    print(f"    {name}  (created {t}, {(latest_time-t).total_seconds()/60:.0f} min before latest)")
                print("  Investigate these specifically before trusting the aggregate.")
            else:
                print("  All matched runs cluster together -- consistent with one coherent launch. OK.")

    rows = []
    for r in candidates:
        hist = r.history(pandas=True, samples=100000)
        if hist is None or len(hist) == 0:
            print(f"  {r.name}: no logged history -- excluding")
            continue
        hist["run_name"] = r.name
        hist["wandb_run_id"] = r.id
        hist["created_at"] = str(r.created_at)
        hist["condition"] = r.config.get("condition")
        hist["seed"] = r.config.get("seed")
        hist["lam_init"] = r.config.get("lam_init")
        hist["alpha"] = r.config.get("alpha")
        rows.append(hist)
        print(f"  {r.name} ({r.id}): {len(hist)} logged steps")
    if not rows:
        raise SystemExit("no usable runs -- check the manifest path or use --source logs")
    return pd.concat(rows, ignore_index=True)


def from_logs(log_dir="ablation_logs"):
    paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    print(f"parsing {len(paths)} log files in {log_dir}/")
    rows = []
    for p in paths:
        with open(p, errors="ignore") as f:
            for line in f:
                m = EVAL_RE.search(line)
                if not m:
                    continue
                d = m.groupdict()
                run = d["run"]
                cond = run.rsplit("_seed", 1)[0]
                seed = int(run.rsplit("_seed", 1)[1])
                rows.append({
                    "run_name": run, "condition": cond, "seed": seed,
                    "iteration": int(d["it"]),
                    "eval/win_rate": float(d["win"]),
                    "eval/sap_mean": float(d["sap"]),
                    "eval/mecha_mean": float(d["mecha"]) if d["mecha"] != "nan" else np.nan,
                    "eval/csi_sap_proxy": (float(d["csi"]) if d["csi"] not in ("None", "") else np.nan),
                    "disc_health": d["health"],
                    "loss/lam": float(d["lam"]),
                    "eval_seconds": int(d["evsec"]),
                })
        print(f"  {os.path.basename(p)}: {sum(1 for r in rows if r['run_name'] in p or True)} cumulative rows")
    if not rows:
        raise SystemExit(f"no EVAL lines matched in {log_dir}/ -- check the directory")
    return pd.DataFrame(rows)


def summarise(df):
    """Final-eval row per run, then per-condition mean/sd across seeds."""
    metric_cols = [c for c in df.columns if c.startswith("eval/")]
    it_col = "iteration" if "iteration" in df.columns else "_step"

    finals = (df.sort_values(it_col)
                .groupby("run_name", as_index=False)
                .last())

    summary = {}
    for cond, grp in finals.groupby("condition"):
        entry = {"n_seeds": int(len(grp)), "seeds": sorted(grp["seed"].tolist())}
        for m in metric_cols:
            if m not in grp:
                continue
            vals = pd.to_numeric(grp[m], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            entry[m] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
        summary[cond] = entry
    return finals, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wandb", "logs"], default="wandb")
    ap.add_argument("--log-dir", default="ablation_logs")
    ap.add_argument("--run-names-file", default=None,
                    help="ablation_parallel_summary.json -- the authoritative list of "
                         "the 37 confirmed-successful run names, used to filter out "
                         "every earlier/duplicate launch attempt in a messy W&B history. "
                         "Strongly recommended whenever run_ablation_parallel.py has been "
                         "launched more than once.")
    args = ap.parse_args()

    df = from_wandb(args.run_names_file) if args.source == "wandb" else from_logs(args.log_dir)
    df.to_csv("recovered_metrics_full.csv", index=False)

    finals, summary = summarise(df)
    finals.to_csv("recovered_metrics_final.csv", index=False)
    with open("recovered_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL EVAL, per condition (mean +/- sd across seeds)")
    print("=" * 78)
    hdr = f"{'condition':22s} {'n':>2s} {'win rate':>16s} {'SAP %':>16s} {'CSI_SAP':>18s}"
    print(hdr); print("-" * 78)
    for cond, e in sorted(summary.items()):
        def fmt(key, prec=4):
            if key not in e:
                return " " * 16
            return f"{e[key]['mean']:.{prec}f} +/- {e[key]['sd']:.{prec}f}"
        print(f"{cond:22s} {e['n_seeds']:>2d} {fmt('eval/win_rate'):>16s} "
              f"{fmt('eval/sap_mean',2):>16s} {fmt('eval/csi_sap_proxy'):>18s}")

    print("\nwrote: recovered_metrics_full.csv, recovered_metrics_final.csv, recovered_summary.json")
    print("\nREFERENCE VALUES for interpretation:")
    print("  frozen baseline win rate : 0.624")
    print("  frozen baseline SAP      : ~87.6%")
    print("  human demonstration SAP  : 17.68%")
    print("  human CSI_SAP target     : -0.316  (i.e. -31.6 percentage points, NEGATIVE)")


if __name__ == "__main__":
    main()
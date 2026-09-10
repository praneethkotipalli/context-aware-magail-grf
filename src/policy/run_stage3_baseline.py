"""
run_stage3_baseline.py

Executes Stage 3 of the MAGAIL-C v2 pipeline.
Launches 8 frozen MAPPO evaluation workers in parallel (500 episodes each).
Aggregates the results into a definitive baseline summary with tight CIs.
Adheres strictly to the v2 storage and MANIFEST discipline.
"""

import os
import sys
import json
import glob
import uuid
import shutil
import datetime
import subprocess
import numpy as np

def main():
    # 1. Generate v2 LAUNCH_ID and Setup Directories
    now = datetime.datetime.now()
    launch_id = f"{now:%m%d}{uuid.uuid4().hex[:3]}_s3base"
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "results_v2", "stage3_baseline", launch_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*60)
    print(f" STAGE 3: MAPPO BASELINE EVALUATION (500 EPISODES)")
    print(f" LAUNCH_ID: {launch_id}")
    print(f" OUT_DIR:   {out_dir}")
    print("="*60)

    # 2. Launch Workers in Parallel
    seeds = list(range(8))
    processes = []
    
    print(f"Launching {len(seeds)} workers in parallel. This will take ~15-20 minutes...")
    
    for seed in seeds:
        log_file = os.path.join(out_dir, f"worker_seed{seed}.log")
        cmd = [
            sys.executable, "finetune_loop.py",
            "--condition", "MAPPO-baseline",
            "--seed", str(seed),
            "--max-iterations", "0",
            "--frozen", "1",
            "--eval-episodes", "500"
        ]
        
        with open(log_file, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            processes.append((seed, p))
            print(f"  -> Started Seed {seed} (PID: {p.pid})")

    # 3. Wait for all to finish
    for seed, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"WARNING: Seed {seed} exited with code {p.returncode}. Check {out_dir}/worker_seed{seed}.log")
        else:
            print(f"  <- Seed {seed} completed successfully.")

    # 4. Gather and Move Result Files
    json_files = glob.glob("result_MAPPO-baseline_seed*.json")
    if not json_files:
        print("ERROR: No result JSON files found. Workers may have failed.")
        sys.exit(1)
        
    data = []
    for jf in json_files:
        dest = os.path.join(out_dir, jf)
        shutil.move(jf, dest)
        with open(dest, "r") as f:
            data.append(json.load(f))

    # 5. Aggregate Metrics
    win_rates = [d["win_rate"] for d in data]
    saps = [d["sap_mean"] for d in data]
    csis = [d["csi_sap_proxy"] for d in data]

    def calc_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "se": float(np.std(arr) / np.sqrt(len(arr)))
        }

    summary = {
        "launch_id": launch_id,
        "stage": "stage3_baseline",
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "n_seeds": len(data),
        "episodes_per_seed": 500,
        "baseline_metrics": {
            "win_rate": calc_stats(win_rates),
            "sap": calc_stats(saps),
            "csi": calc_stats(csis)
        }
    }

    summary_path = os.path.join(out_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Append to Global MANIFEST
    manifest_path = os.path.join(project_root, "results_v2", "MANIFEST.json")
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    manifest_entry = {
        "launch_id": launch_id,
        "stage": "stage3_baseline",
        "utc": summary["utc"],
        "git_sha": git_sha,
        "n_runs": len(data),
        "notes": "Stage 3 MAPPO Baseline 500-episode Evaluation"
    }

    if os.path.exists(manifest_path):
        with open(manifest_path, "r+") as f:
            try:
                manifest_data = json.load(f)
            except json.JSONDecodeError:
                manifest_data = []
            if not isinstance(manifest_data, list):
                manifest_data = [manifest_data]
            manifest_data.append(manifest_entry)
            f.seek(0)
            json.dump(manifest_data, f, indent=2)
            f.truncate()
    else:
        with open(manifest_path, "w") as f:
            json.dump([manifest_entry], f, indent=2)

    # 7. Print Final Results
    print("\n" + "="*60)
    print(" BASELINE AGGREGATION COMPLETE")
    print("="*60)
    print(f"Win Rate: {summary['baseline_metrics']['win_rate']['mean']:.4f} ± {summary['baseline_metrics']['win_rate']['se']:.4f}")
    print(f"SAP:      {summary['baseline_metrics']['sap']['mean']:.2f}% ± {summary['baseline_metrics']['sap']['se']*100:.2f}%")
    print(f"CSI:      {summary['baseline_metrics']['csi']['mean']:+.4f} ± {summary['baseline_metrics']['csi']['se']:.4f}")
    print(f"\nResults saved to: {summary_path}")

if __name__ == "__main__":
    main()
"""
final_evaluation.py

Evaluates saved ablation checkpoints at a chosen episode count, in
parallel, logging to BOTH W&B and local JSON.

Two modes:
  --mode worker   evaluates ONE checkpoint (invoked by the launcher)
  --mode launch   spawns one worker process per checkpoint, up to --workers

WHY SEPARATE OS PROCESSES, not a loop or threads:
ContextConditionedActor.__init__ does `self.base = frozen_actor.base` -- a
SHARED REFERENCE, not a copy. Building several actors from one loaded
frozen_actor makes them share base/out weights, and load_state_dict on one
silently mutates the others. A sequential loop survives this by accident
(each is evaluated before the next load); anything concurrent in-process
would not. Separate processes make the isolation structural.

TWO USES:
  1. Patch missing metrics into result_*.json  (--episodes 50)
  2. Final high-precision evaluation for figures (--episodes 500)

ON "BEST CHECKPOINT": do NOT select the best seed per condition. The locked
protocol is 5 seeds per condition with Mann-Whitney across seeds; picking
the best seed is cherry-picking and invalidates the statistics. This script
evaluates ALL seeds. Selecting the best LAMBDA from the sweep IS legitimate
-- that is what the sweep exists for (Section 3.5.2).
"""

import argparse, glob, json, os, subprocess, sys, time

import torch
torch.set_num_threads(1)   # mandatory when running many workers concurrently

PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")
sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.join(GRF_MARL_ROOT, "light_malib", "model",
                                 "gr_football", "enhanced_LightActionMask_5"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "policy"))

ACTOR_PATH = os.path.join(GRF_MARL_ROOT,
    "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")

# reference values for interpreting output
BASELINE_WIN_RATE = 0.624
BASELINE_SAP = 87.64
HUMAN_SAP = 17.68
HUMAN_CSI_SAP = -0.316


def run_worker(checkpoint, episodes, out_dir, wandb_project, frozen):
    import numpy as np
    import gfootball.env as football_env
    from enhanced_LightActionMask_5 import FeatureEncoder
    from context_conditioned_policy import ContextConditionedActor
    from evaluate_policy import evaluate_policy

    def build_env():
        return football_env.create_environment(
            env_name="5_vs_5_d06", representation="raw",
            number_of_left_players_agent_controls=4,
            number_of_right_players_agent_controls=0, render=False)

    if frozen:
        run_name, condition, seed = "MAPPO-baseline_frozen", "MAPPO-baseline", 0
    else:
        run_name = os.path.basename(checkpoint).replace("ablation_", "").replace(".pt", "")
        condition, seed = run_name.rsplit("_seed", 1)
        seed = int(seed)

    # each worker loads its OWN frozen actor -- see module docstring
    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor)

    if not frozen:
        ck = torch.load(checkpoint, map_location="cpu")
        actor.load_state_dict(ck["actor"])
        iters = ck.get("iterations_completed")
        halted = ck.get("halted")
    else:
        # context_net is zero-initialised, so this reproduces the raw
        # checkpoint's behaviour exactly (verified in test_policy_pilot.py)
        iters, halted = 0, False
    actor.eval()

    try:
        import wandb
        wandb.init(project=wandb_project, name=f"finaleval_{run_name}", reinit=True,
                   config={"condition": condition, "seed": seed, "episodes": episodes,
                           "checkpoint": os.path.basename(checkpoint) if not frozen else "frozen",
                           "eval_type": "final"})
        use_wandb = True
    except Exception as e:
        print(f"  wandb unavailable ({e}) -- local JSON only")
        use_wandb = False

    t0 = time.time()
    ev = evaluate_policy(actor, FeatureEncoder(), build_env,
                         n_episodes=episodes, max_steps=3000)
    elapsed = time.time() - t0

    result = {
        "run_name": run_name, "condition": condition, "seed": seed,
        "episodes": episodes, "eval_seconds": elapsed,
        "iterations_completed": iters, "halted_during_training": halted,
        **{k: v for k, v in ev.items()},
        "reference": {"baseline_win_rate": BASELINE_WIN_RATE,
                      "baseline_sap": BASELINE_SAP,
                      "human_sap": HUMAN_SAP,
                      "human_csi_sap": HUMAN_CSI_SAP},
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"finaleval_{run_name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    if use_wandb:
        wandb.log({f"final/{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"{run_name}: win={ev['win_rate']:.3f}  SAP={ev['sap_mean']:.2f}%  "
          f"MECHA={ev['mecha_mean']:.4f}  CSI={ev['csi_sap_proxy']}  ({elapsed/60:.1f}min)")
    return result


def launch(args):
    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "ablation_*.pt")))
    if args.conditions:
        ckpts = [c for c in ckpts
                 if any(f"ablation_{cond}_seed" in os.path.basename(c) for cond in args.conditions)]

    jobs = [(c, False) for c in ckpts]
    # the frozen branch of finetune_loop.run() returns before saving, so no
    # MAPPO-baseline checkpoint exists -- evaluate the frozen policy directly
    if not args.conditions or "MAPPO-baseline" in args.conditions:
        jobs.append((None, True))

    print(f"{len(jobs)} evaluation jobs, {args.episodes} episodes each, "
          f"{args.workers} concurrent")
    for c, fr in jobs:
        print(f"  {'FROZEN BASELINE' if fr else os.path.basename(c)}")
    if args.dry_run:
        print("\nDRY RUN -- nothing launched.")
        return

    os.makedirs(args.log_dir, exist_ok=True)
    queue, running, done = list(jobs), [], []
    t_start = time.time()

    while queue or running:
        while queue and len(running) < args.workers:
            c, fr = queue.pop(0)
            name = "MAPPO-baseline_frozen" if fr else \
                   os.path.basename(c).replace("ablation_", "").replace(".pt", "")
            logf = open(os.path.join(args.log_dir, f"{name}.log"), "w")
            cmd = [sys.executable, os.path.abspath(__file__), "--mode", "worker",
                   "--episodes", str(args.episodes), "--out-dir", args.out_dir,
                   "--wandb-project", args.wandb_project]
            cmd += ["--frozen"] if fr else ["--checkpoint", c]
            running.append({"name": name, "proc": subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT), "logf": logf,
                "start": time.time()})
            print(f"  launched {name} ({len(running)}/{args.workers}, {len(queue)} queued)")

        time.sleep(5)
        still = []
        for j in running:
            if j["proc"].poll() is None:
                still.append(j)
            else:
                j["logf"].close()
                el = time.time() - j["start"]
                print(f"  finished {j['name']} ({el/60:.1f}min, "
                      f"{'OK' if j['proc'].returncode == 0 else f'RC={j[chr(39)+chr(39)]}'})"
                      if False else
                      f"  finished {j['name']} ({el/60:.1f}min, rc={j['proc'].returncode})")
                done.append({"name": j["name"], "returncode": j["proc"].returncode,
                             "wall_sec": el})
        running = still

    print(f"\nall done in {(time.time()-t_start)/3600:.2f}h")
    summarise(args.out_dir)


def summarise(out_dir):
    import numpy as np
    files = sorted(glob.glob(os.path.join(out_dir, "finaleval_*.json")))
    if not files:
        print("no result files found"); return
    rows = [json.load(open(f)) for f in files]

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    print("\n" + "=" * 92)
    print(f"FINAL EVALUATION  ({rows[0]['episodes']} episodes per run)")
    print("=" * 92)
    print(f"{'condition':20s} {'n':>2s} {'win rate':>15s} {'SAP %':>15s} "
          f"{'MECHA':>15s} {'CSI_SAP':>17s}")
    print("-" * 92)
    summary = {}
    for cond, rs in sorted(by_cond.items()):
        entry = {"n_seeds": len(rs)}
        cells = []
        for key in ("win_rate", "sap_mean", "mecha_mean", "csi_sap_proxy"):
            v = np.array([r[key] for r in rs if r.get(key) is not None], dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                cells.append(" " * 15); continue
            m, s = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
            entry[key] = {"mean": m, "sd": s, "values": v.tolist()}
            cells.append(f"{m:.4f}±{s:.4f}")
        summary[cond] = entry
        print(f"{cond:20s} {len(rs):>2d} {cells[0]:>15s} {cells[1]:>15s} "
              f"{cells[2]:>15s} {cells[3]:>17s}")

    with open(os.path.join(out_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nREFERENCE  baseline win 0.624 | baseline SAP 87.64% | "
          "human SAP 17.68% | human CSI_SAP -0.316")
    print(f"wrote {out_dir}/final_summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["worker", "launch", "summarise"], default="launch")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--frozen", action="store_true")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--workers", type=int, default=25)
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--out-dir", default="final_eval_results")
    p.add_argument("--log-dir", default="final_eval_logs")
    p.add_argument("--wandb-project", default="magail-c-finaleval")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="e.g. MAGAIL-C MAGAIL-SC MAGAIL-C+KL MAGAIL-NC MAPPO-baseline")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.mode == "worker":
        run_worker(args.checkpoint, args.episodes, args.out_dir,
                   args.wandb_project, args.frozen)
    elif args.mode == "summarise":
        summarise(args.out_dir)
    else:
        launch(args)


if __name__ == "__main__":
    main()
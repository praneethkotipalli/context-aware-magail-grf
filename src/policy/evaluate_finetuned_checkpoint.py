"""
evaluate_finetuned_checkpoint.py -- v2, adds W&B logging.

finetune_loop.py's --frozen mode always evaluates the ORIGINAL,
unfine-tuned MAPPO checkpoint (hardcoded to ACTOR_PATH) -- it has no way
to evaluate a SAVED fine-tuned actor. This fills that gap: loads an
ablation_{run_name}.pt checkpoint (produced at the end of a real
fine-tuning run) and runs a real N-episode evaluation, matching the
frozen baseline's own 500-episode standard so the numbers are directly
comparable.

Each call logs one W&B run under project "magail-c-v2-finaleval", grouped
by --group (the condition label: NC/C/SC/C+KL) so the 32 runs can be
filtered/grouped in the W&B UI directly, without opening any JSON.

Run:  python evaluate_finetuned_checkpoint.py \
          --checkpoint ablation_final-C_seed0.pt \
          --episodes 500 --out final_eval_final-C_seed0.json --group C
"""
import argparse, json, time
import torch
import wandb

from finetune_loop import build_env, ACTOR_PATH, MAX_STEPS
from context_conditioned_policy import ContextConditionedActor
from evaluate_policy import evaluate_policy
from enhanced_LightActionMask_5 import FeatureEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default=None,
                    help="condition label (NC/C/SC/C+KL) for W&B grouping")
    ap.add_argument("--wandb-project", default="magail-c-v2-finaleval")
    args = ap.parse_args()

    # ContextConditionedActor wraps a base module of the SAME shape as the
    # original frozen checkpoint -- must build with that structure first,
    # then overwrite weights with the fine-tuned state_dict.
    frozen_actor_structure = torch.load(ACTOR_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor_structure)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    seed = ckpt.get("seed")
    run_name = f"finaleval_{args.group or 'unknown'}_seed{seed}"

    wandb.init(project=args.wandb_project, name=run_name, group=args.group,
              reinit=True, config={
                  "checkpoint": args.checkpoint, "episodes": args.episodes,
                  "condition_full": ckpt.get("condition"), "condition_label": args.group,
                  "seed": seed, "disc_int_condition": ckpt.get("disc_int_condition"),
              })

    encoder = FeatureEncoder()
    t0 = time.time()
    ev = evaluate_policy(actor, encoder, build_env, n_episodes=args.episodes, max_steps=MAX_STEPS)
    elapsed = time.time() - t0

    result = {"checkpoint": args.checkpoint, "episodes": args.episodes,
              "wall_sec": elapsed, "condition": ckpt.get("condition"),
              "condition_label": args.group, "seed": seed,
              "disc_int_condition": ckpt.get("disc_int_condition"), **ev}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    wandb.log({f"final500/{k}": v for k, v in ev.items() if isinstance(v, (int, float))})
    wandb.finish()

    print(f"wrote {args.out}: win={ev['win_rate']:.3f} SAP={ev['sap_mean']:.2f}% "
          f"({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
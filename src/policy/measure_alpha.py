"""
measure_alpha.py -- v2, updated for the post-Stage-1 finetune_loop.py.

Two fixes vs the version this replaces:
  1. collect_rollout() no longer takes use_context -- [B2] removed it
     entirely, policy is always context-conditioned now. Call with no
     kwarg.
  2. D_INT_CKPT points at D_int_C_production.pt (the new naming from
     train_D_int_variant.py --variant C), not the retired
     D_int_production_tiny-interact.pt.

Everything else (raw-sum-parity calibration, GATE B check) is unchanged.

Run:  python measure_alpha.py --episodes 5
"""
import argparse
import numpy as np
import torch

from finetune_loop import (collect_rollout, build_env, ACTOR_PATH, CRITIC_PATH,
                           DISC_CKPT, NORMALISER_PATH, PROJECT_ROOT)
from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from finetune_objective import compute_style_reward
from feature_normalization import FeatureNormaliser
from enhanced_LightActionMask_5 import FeatureEncoder
from discriminator_model import Discriminator as MargDiscriminator

import sys, os
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "discriminator"))
from discriminator_candidates import TinyInteract

D_INT_CKPT = os.path.join(PROJECT_ROOT, "src", "discriminator",
                          "D_int_C_production.pt")
TARGET_RATIO = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
    frozen_critic = torch.load(CRITIC_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor); actor.eval()
    critic = ContextConditionedCritic(frozen_critic); critic.eval()

    D_marg = MargDiscriminator()
    D_marg.load_state_dict(torch.load(DISC_CKPT, map_location="cpu")["model_state_dict"])
    D_marg.eval()

    D_int = TinyInteract()
    D_int.load_state_dict(torch.load(D_INT_CKPT, map_location="cpu")["model_state_dict"])
    D_int.eval()
    print(f"D_marg <- {DISC_CKPT}")
    print(f"D_int  <- {D_INT_CKPT}\n")

    normaliser = FeatureNormaliser(); normaliser.load(NORMALISER_PATH)
    encoder = FeatureEncoder()
    env = build_env()

    task_sums, marg_sums, int_sums = [], [], []
    task_stds, marg_stds, int_stds = [], [], []
    n_steps_total = 0

    for ep in range(args.episodes):
        batch = collect_rollout(actor, critic, encoder, env, normaliser)
        feats = torch.as_tensor(batch["disc_feat"], dtype=torch.float32)
        n_steps_total += batch["n_steps"]

        r_marg = compute_style_reward(D_marg, feats)
        r_int = compute_style_reward(D_int, feats)
        r_task = batch["rewards"]

        task_sums.append(r_task.abs().sum().item())
        marg_sums.append(r_marg.abs().sum().item())
        int_sums.append(r_int.abs().sum().item())
        task_stds.append(r_task.std().item())
        marg_stds.append(r_marg.std().item())
        int_stds.append(r_int.std().item())

        print(f"  ep{ep}: steps={batch['n_steps']:4d}  "
              f"sum|task|={task_sums[-1]:7.2f}  sum|marg|={marg_sums[-1]:7.2f}  "
              f"sum|int|={int_sums[-1]:7.2f}")

    env.close()

    task_tot, marg_tot, int_tot = sum(task_sums), sum(marg_sums), sum(int_sums)
    marg_ratio = marg_tot / max(task_tot, 1e-9)
    int_ratio = int_tot / max(task_tot, 1e-9)

    print(f"\n{'='*64}\nGATE B -- raw summed-reward-magnitude ratios (pre-GAE, pre-alpha)\n{'='*64}")
    print(f"  sum|task| total = {task_tot:.2f}  (over {n_steps_total} steps, "
          f"{args.episodes} episodes)")
    print(f"  marg ratio (sum|r_marg| / sum|task|) = {marg_ratio:8.2f}x   "
          f"(std ratio: {np.mean(marg_stds)/max(np.mean(task_stds),1e-9):.2f}x)")
    print(f"  int  ratio (sum|r_int|  / sum|task|) = {int_ratio:8.2f}x   "
          f"(std ratio: {np.mean(int_stds)/max(np.mean(task_stds),1e-9):.2f}x)")

    alpha_marg = TARGET_RATIO / max(marg_ratio, 1e-9)
    alpha_int = TARGET_RATIO / max(int_ratio, 1e-9)
    print(f"\n  suggested ALPHA_MARG (targets {TARGET_RATIO}x raw-sum parity) = {alpha_marg:.6f}")
    print(f"  suggested ALPHA_INT  (targets {TARGET_RATIO}x raw-sum parity) = {alpha_int:.6f}")

    ok = marg_ratio <= 10 and int_ratio <= 10
    print(f"\n  {'PASS' if ok else 'FAIL'}: both ratios "
          f"{'already within' if ok else 'NOT within'} 10x of task at alpha=1.0")
    if not ok:
        print("  Confirms the mechanism behind v1's 27-point win-rate collapse.")
    print(f"  Compare these alpha_marg/alpha_int against finetune_loop.py's current")
    print(f"  ALPHA_MARG=0.0048 / ALPHA_INT=0.0026 -- should be close.")


if __name__ == "__main__":
    main()

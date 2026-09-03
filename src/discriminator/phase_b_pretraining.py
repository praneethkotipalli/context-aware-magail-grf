"""
phase_b_pretraining.py -- Phase B (3.3.3): continued pre-training vs.
frozen-MAPPO, resuming from the Phase A checkpoint (not fresh init).

Reuses the SAME held-out split (identical seed) as Phase A -- testing
generalization against a fixed, consistent set of unseen episodes across
both phases, not a different random draw each time.
"""

import numpy as np
import torch
import wandb

from discriminator_model import Discriminator
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from held_out_split import make_episode_split, split_features_by_episode
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
BATCH_SIZE = 128
N_STEPS = 10_000
LOG_EVERY = 100
EVAL_EVERY = 500
HELD_OUT_FRAC = 0.20
SEED = 0                                    # SAME seed as Phase A -- identical held-out episodes
PHASE_A_CHECKPOINT = "discriminator_phase_a_checkpoint.pt"
CHECKPOINT_PATH = "discriminator_phase_b_checkpoint.pt"
PASS_THRESHOLD = 0.75                       # locked, Section 3.3.3 -- combined-phase gate


def compute_held_out_accuracy(model, held_feat, held_bins, batch_size=512):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(held_feat), batch_size):
            batch = torch.as_tensor(held_feat[i:i+batch_size], dtype=torch.float32)
            probs = model.probability(batch)
            correct += (probs > 0.5).sum().item()
            total += len(batch)
    model.train()
    return correct / total


def run():
    rng = np.random.default_rng(SEED)
    wandb.init(project="magail-c", name="phase_b_pretraining", config={
        "batch_size": BATCH_SIZE, "n_steps": N_STEPS, "phase": "B_frozen_mappo",
        "eta": 0.3, "held_out_frac": HELD_OUT_FRAC, "seed": SEED,
        "resumed_from": PHASE_A_CHECKPOINT,
    })

    expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
    train_ep, held_ep = make_episode_split(
        expert_cache["episode_outcomes"], held_out_frac=HELD_OUT_FRAC, seed=SEED
    )
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
    )
    expert_sampler = BalancedContextSampler(train_feat, train_bins, name="expert_train")
    target_props = sqrt_scaled_target(expert_sampler.cell_counts)

    agent_cache = np.load("mappo_features_cache.npz")
    agent_sampler = BalancedContextSampler(agent_cache["features"], agent_cache["bins"], name="frozen_mappo")

    model = Discriminator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)
    mask_lw = select_by_true_context(
        train_feat, t_norm_max=LATE_WINNING['t_norm_max'], delta_score_sign=LATE_WINNING['delta_score_sign']
    )
    mask_ll = select_by_true_context(
        train_feat, t_norm_max=LATE_LOSING['t_norm_max'], delta_score_sign=LATE_LOSING['delta_score_sign']
    )
    checkpoint = torch.load(PHASE_A_CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed from {PHASE_A_CHECKPOINT} (Phase A step {checkpoint['step']})")

    trainer = DiscriminatorTrainer(
        model, 
        optimizer, 
        expert_sampler=expert_sampler, 
        eta=0.3,                               # Lowered R1 penalty
        gamma_swap=6.0,                        # Heavy context pressure
        swap_lw_features=train_feat[mask_lw],  # Activates the LW swap loss
        swap_ll_features=train_feat[mask_ll]   # Activates the LL swap loss
    )
    print(f"\nPhase B: {N_STEPS} steps vs. frozen-MAPPO, batch_size={BATCH_SIZE}\n")

    for step in range(1, N_STEPS + 1):
        (e_feat, e_bins), (a_feat, a_bins) = balanced_batch(
            expert_sampler, agent_sampler, batch_size=BATCH_SIZE,
            target_props=target_props, rng=rng, on_empty='skip',
        )
        result = trainer.step(e_feat, a_feat, expert_cells=e_bins, agent_cells=a_bins)

        if step % LOG_EVERY == 0:
            wandb.log({
                "step": step, "loss": result["loss"],
                "bce_expert": result["bce_expert"], "bce_agent": result["bce_agent"],
                "acc": result["acc"], "acc_expert": result["acc_expert"], "acc_agent": result["acc_agent"],
                "gp_value": result["gp_value"], "ess": result["ess"],
            })
            print(f"  step {step}: loss={result['loss']:.4f}  acc={result['acc']:.3f}  "
                  f"gp={result['gp_value']:.4f}  health={trainer.health()}")

        if step % EVAL_EVERY == 0 or step == N_STEPS:
            held_acc = compute_held_out_accuracy(model, held_feat, held_bins)
            wandb.log({"step": step, "held_out_acc": held_acc})
            print(f"    [eval] held-out accuracy: {held_acc:.4f}")

    final_held_acc = compute_held_out_accuracy(model, held_feat, held_bins)
    combined_pass = final_held_acc >= PASS_THRESHOLD

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": N_STEPS, "phase": "B",
        "final_held_out_acc": final_held_acc,
    }, CHECKPOINT_PATH)

    print(f"\nPhase B complete. Checkpoint saved to {CHECKPOINT_PATH}")
    print(f"COMBINED (Phase A+B) held-out accuracy: {final_held_acc:.4f}")
    print(f"COMBINED GATE: {'PASS' if combined_pass else 'FAIL'} (threshold {PASS_THRESHOLD})")
    print("Next: run the counterfactual swap-test gate against this checkpoint for the real verdict.")
    wandb.finish()


if __name__ == '__main__':
    run()
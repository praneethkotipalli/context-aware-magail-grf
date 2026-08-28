"""
check_swap_loss_magnitude.py

Calibrates gamma_swap the same way eta and beta_aux were calibrated --
against REAL data this time (train split), not synthetic noise, since the
swap-consistency term's magnitude depends on how far current sensitivity
already is from swap_margin, which is a real, not synthetic, quantity.

Run this AFTER building swap_lw_features/swap_ll_features from the train
split (same code phase_a_pretraining.py will use), BEFORE a full retrain.
"""
import numpy as np
import torch

from discriminator_model import Discriminator
from discriminator_loss import DiscriminatorLoss, DEFAULT_GAMMA_SWAP
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode

expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)
(train_feat, train_bins), (_, _) = split_features_by_episode(
    expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
)

mask_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING['t_norm_max'], delta_score_sign=LATE_WINNING['delta_score_sign'])
mask_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING['t_norm_max'], delta_score_sign=LATE_LOSING['delta_score_sign'])
swap_lw_features = train_feat[mask_lw]
swap_ll_features = train_feat[mask_ll]
print(f"TRAIN-split late-winning population: {swap_lw_features.shape[0]}")
print(f"TRAIN-split late-losing population:  {swap_ll_features.shape[0]}")

# Load the CURRENT checkpoint (the one that just failed) -- calibrating
# against where the network ACTUALLY is right now, not a fresh init,
# since that's what the next training run will actually resume from.
model = Discriminator()
ckpt = torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])

expert_batch = torch.as_tensor(train_feat[np.random.choice(len(train_feat), 128)], dtype=torch.float32)
agent_cache = np.load("mappo_features_cache.npz")
agent_batch = torch.as_tensor(agent_cache["features"][np.random.choice(len(agent_cache["features"]), 128)], dtype=torch.float32)

print(f"\n{'gamma_swap':>10s}  {'bce_total':>10s}  {'r1':>8s}  {'swap_lw':>10s}  {'swap_ll':>10s}  {'total_swap':>11s}  {'swap_share':>11s}")
for gamma in [0.0, 0.5, DEFAULT_GAMMA_SWAP, 3.0, 5.0]:
    loss_fn = DiscriminatorLoss(
        gamma_swap=gamma, swap_lw_features=swap_lw_features, swap_ll_features=swap_ll_features,
    )
    out = loss_fn(model, expert_batch, agent_batch)
    bce_total = out.bce_expert.item() + out.bce_agent.item()
    r1 = out.r1_penalty.item()
    hinge_lw = out.swap_hinge_lw.item() if out.swap_hinge_lw is not None else 0.0
    hinge_ll = out.swap_hinge_ll.item() if out.swap_hinge_ll is not None else 0.0
    total_swap_contribution = gamma * (hinge_lw + hinge_ll)
    total = bce_total + r1 + total_swap_contribution
    share = total_swap_contribution / total * 100 if total > 0 else 0
    print(f"{gamma:>10.2f}  {bce_total:>10.4f}  {r1:>8.4f}  {hinge_lw:>10.4f}  {hinge_ll:>10.4f}  "
          f"{total_swap_contribution:>11.4f}  {share:>10.1f}%")

print(f"\nACTUAL current mean|shift| (unscaled, informational):")
print(f"  late_winning -> late_losing: {out.swap_shift_lw.item() if out.swap_shift_lw is not None else 'N/A'}")
print(f"  late_losing -> late_winning: {out.swap_shift_ll.item() if out.swap_shift_ll is not None else 'N/A'}")
print(f"  (compare against last night's gate numbers: 0.0072 / 0.0648 -- should roughly match)")
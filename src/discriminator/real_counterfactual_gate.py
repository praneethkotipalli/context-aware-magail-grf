"""run_real_counterfactual_gate.py -- first genuine verdict, not a smoke test."""
import numpy as np
import torch

from discriminator_model import Discriminator
from counterfactual_gate import run_counterfactual_gate, print_gate_report
from held_out_split import make_episode_split, split_features_by_episode

SEED = 0  # same seed as both pre-training phases -- same held-out episodes

expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=SEED)
(_, _), (held_feat, held_bins) = split_features_by_episode(
    expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
)
print(f"Running gate against HELD-OUT expert data only ({held_feat.shape[0]} steps) -- "
      f"genuinely unseen by this discriminator, consistent with keeping held-out data reserved for evaluation.")

model = Discriminator()
checkpoint = torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded Phase B checkpoint (final held-out acc: {checkpoint['final_held_out_acc']:.4f})\n")

overall_pass, summary = run_counterfactual_gate(model, held_feat)
print_gate_report(overall_pass, summary)

if overall_pass:
    print("\nGate PASSED -- discriminator is context-sensitive in both directions.")
    print("Safe to proceed to policy-side integration.")
else:
    print("\nGate FAILED -- do NOT connect this discriminator to the policy.")
    print("Diagnose before proceeding: check warnings above for which direction/population failed.")
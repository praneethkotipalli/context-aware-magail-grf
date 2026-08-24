"""diagnose_context_sensitivity.py"""
import numpy as np, torch
from discriminator_model import Discriminator
from feature_derivation import BLOCK_SLICES
from held_out_split import make_episode_split, split_features_by_episode
from counterfactual_gate import run_counterfactual_gate

cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(cache["episode_outcomes"], 0.20, seed=0)
(train_feat, _), (held_feat, _) = split_features_by_episode(
    cache["features"], cache["bins"], cache["episode_ids"], train_ep, held_ep)

model = Discriminator()
model.load_state_dict(torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")["model_state_dict"])
model.eval()

# 1. Per-block input-gradient magnitude -- is context under-weighted vs other blocks?
x = torch.as_tensor(held_feat[:2000], dtype=torch.float32).requires_grad_(True)
logits = model(x)
grads = torch.autograd.grad(logits.sum(), x)[0].abs().mean(dim=0)
print("Mean |d(logit)/d(input)| per block, PER DIMENSION:")
for name, sl in BLOCK_SLICES.items():
    per_dim = grads[sl].mean().item()
    print(f"  {name:10s} ({sl.stop-sl.start:3d} dims): {per_dim:.5f}")
print(f"\n  overall per-dim mean: {grads.mean().item():.5f}")
ctx = grads[BLOCK_SLICES['context']]
print(f"  context dims individually: T_norm={ctx[0].item():.5f}  dScore={ctx[1].item():.5f}")

# 2. First-layer weight norms -- did the network allocate capacity to context at all?
W = model.net[0].weight.detach()
print("\nFirst-layer incoming weight norm per input dim (mean per block):")
for name, sl in BLOCK_SLICES.items():
    print(f"  {name:10s}: {W[:, sl].norm(dim=0).mean().item():.5f}")

# 3. Does the gate pass on TRAIN data? (learned-but-not-generalizing vs never-learned)
print("\n--- gate on TRAIN data (was it ever learned?) ---")
overall, summary = run_counterfactual_gate(model, train_feat)
for r in summary["directions"]:
    print(f"  {r['direction']}: mean|shift|={r['mean_abs_shift']:.5f}  n={r['n']}")
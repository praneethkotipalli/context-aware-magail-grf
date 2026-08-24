"""diagnose_context_usage.py -- is context actually unused, and how unused?"""
import numpy as np
import torch

from discriminator_model import Discriminator
from held_out_split import make_episode_split, split_features_by_episode

model = Discriminator()
ckpt = torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- 1. First-layer weight magnitude: context dims vs everything else ---
W = model.net[0].weight.detach()          # (256, 137)
per_dim = W.abs().mean(dim=0)             # mean |weight| per input dim
ctx = per_dim[135:137]
spatial = per_dim[:135]

print("First-layer mean |weight| per input dim:")
print(f"  context dims (135,136): {ctx[0]:.5f}, {ctx[1]:.5f}")
print(f"  spatial/action dims   : mean={spatial.mean():.5f}  "
      f"min={spatial.min():.5f}  max={spatial.max():.5f}")
print(f"  context rank among all 137 dims: "
      f"{(per_dim > ctx.max()).sum().item() + 1} of 137 (1 = largest)")

# --- 2. Ablation: does randomizing context change accuracy at all? ---
cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(cache["episode_outcomes"], 0.20, seed=0)
(_, _), (held_feat, _) = split_features_by_episode(
    cache["features"], cache["bins"], cache["episode_ids"], train_ep, held_ep
)

def acc(feats):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(feats), 512):
            b = torch.as_tensor(feats[i:i+512], dtype=torch.float32)
            correct += (model.probability(b) > 0.5).sum().item()
    return correct / len(feats)

true_acc = acc(held_feat)

shuffled = held_feat.copy()
rng = np.random.default_rng(0)
perm = rng.permutation(len(shuffled))
shuffled[:, 135:137] = shuffled[perm, 135:137]   # destroy context-behaviour pairing
shuf_acc = acc(shuffled)

zeroed = held_feat.copy()
zeroed[:, 135:137] = 0.0
zero_acc = acc(zeroed)

print(f"\nHeld-out expert accuracy:")
print(f"  true context     : {true_acc:.4f}")
print(f"  shuffled context : {shuf_acc:.4f}  (delta {shuf_acc-true_acc:+.4f})")
print(f"  zeroed context   : {zero_acc:.4f}  (delta {zero_acc-true_acc:+.4f})")
print("\nIf all three are ~identical, context contributes nothing -- confirming")
print("the gate result, and meaning MAGAIL-C == MAGAIL-SC == MAGAIL-NC by construction.")
"""
diagnose_context_pred_vs_swap.py

Does the context-prediction head actually read the injected context, or
infer it from content (e.g. sticky_actions) and ignore the context slot
entirely? If the latter, that explains why the aux loss dropped (probably)
without the gate moving at all -- the shortcut satisfies the auxiliary
objective without ever exercising film1/film2.
"""
import numpy as np
import torch

from discriminator_model import Discriminator
from context_swap import swap_context
from context_shift_scoring import select_by_true_context, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode

expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)
(_, _), (held_feat, held_bins) = split_features_by_episode(
    expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
)

model = Discriminator()
ckpt = torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mask = select_by_true_context(held_feat, t_norm_max=LATE_LOSING['t_norm_max'],
                               delta_score_sign=LATE_LOSING['delta_score_sign'])
subset = held_feat[mask][:200]
swapped = np.stack([
    swap_context(row, t_norm=0.1, delta_score_raw=+2) for row in subset
])
with torch.no_grad():
    true_t = torch.as_tensor(subset, dtype=torch.float32)
    swap_t = torch.as_tensor(swapped, dtype=torch.float32)
    _, true_ctx_pred = model.forward_with_context_pred(true_t)
    _, swap_ctx_pred = model.forward_with_context_pred(swap_t)

print("TRUE context actual:     ", true_t[:, -2:].mean(dim=0).numpy())
print("TRUE context predicted:  ", true_ctx_pred.mean(dim=0).numpy())
print()
print("SWAPPED context WRITTEN: ", swap_t[:, -2:].mean(dim=0).numpy(), " (target: ~[0.10, +0.667])")
print("SWAPPED context predicted:", swap_ctx_pred.mean(dim=0).numpy())
print()
print("If 'swapped predicted' tracks the WRITTEN target -> head reads context correctly,")
print("problem is elsewhere (film gates decoupled from the prediction path).")
print("If 'swapped predicted' stays near the TRUE ORIGINAL value -> head ignores the")
print("injected context, infers it from content instead -- confirms the shortcut hypothesis.")
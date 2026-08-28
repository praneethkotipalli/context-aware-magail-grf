"""
diagnose_logit_vs_prob_shift.py

Is the gate failing because context ISN'T USED, or because the
discriminator is so confident that the sigmoid crushes real logit
movement into tiny probability movement?

dP/dlogit = P(1-P). At P=0.97 that's 0.029, so a 0.1 probability shift
requires a ~3.4 logit shift. At P=0.75 it's 0.19 -- only ~0.53 needed.
"""
import numpy as np
import torch

from discriminator_model import Discriminator
from context_swap import swap_context
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode

expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)
(_, _), (held_feat, _) = split_features_by_episode(
    expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
)

model = Discriminator()
ckpt = torch.load("discriminator_phase_b_checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def analyze(name, mask, target_t_norm, target_delta):
    subset = held_feat[mask]
    swapped = np.stack([swap_context(row, target_t_norm, target_delta) for row in subset])
    with torch.no_grad():
        tl = model(torch.as_tensor(subset, dtype=torch.float32)).view(-1)
        sl = model(torch.as_tensor(swapped, dtype=torch.float32)).view(-1)
    tp, sp = torch.sigmoid(tl), torch.sigmoid(sl)

    dlogit = (sl - tl).abs()
    dprob = (sp - tp).abs()
    slope = tp * (1 - tp)   # dP/dlogit at the operating point

    print(f"\n{name}  (n={len(subset)})")
    print(f"  mean P(expert) at true context : {tp.mean():.4f}")
    print(f"  P quantiles [10/50/90]         : "
          f"{tp.quantile(torch.tensor([0.1,0.5,0.9])).numpy().round(4)}")
    print(f"  mean sigmoid slope P(1-P)      : {slope.mean():.5f}")
    print(f"  mean |delta logit|             : {dlogit.mean():.4f}")
    print(f"  mean |delta prob|              : {dprob.mean():.4f}   <- what the gate measures")
    print(f"  logit shift needed for dP=0.1  : {0.1/slope.mean():.2f}  "
          f"(currently achieving {dlogit.mean():.2f})")

analyze("late_winning -> late_losing",
        select_by_true_context(held_feat, t_norm_max=LATE_WINNING['t_norm_max'],
                               delta_score_sign=LATE_WINNING['delta_score_sign']),
        0.1, -2)

analyze("late_losing -> late_winning",
        select_by_true_context(held_feat, t_norm_max=LATE_LOSING['t_norm_max'],
                               delta_score_sign=LATE_LOSING['delta_score_sign']),
        0.1, +2)

print("\n" + "="*70)
print("READING THIS:")
print("  |delta logit| > 1.0 but |delta prob| tiny  -> SATURATION is the")
print("     binding constraint. Context IS being used; the sigmoid is")
print("     compressing it. Fix confidence (label smoothing, early stop),")
print("     not architecture.")
print("  |delta logit| also small (< 0.2)          -> context genuinely")
print("     underused at the logit level. Saturation hypothesis is wrong,")
print("     and the problem is upstream of the sigmoid.")
print("="*70)
"""
diagnose_context_separability.py

THE decisive question: how distinguishable is human behaviour in
late-winning vs late-losing states, using ONLY content features?

With a context-independent agent (verified: 4.2pp agent sprint spread),
the optimal context-conditional discriminator's swap-induced logit shift
equals log p_E(x|c') - log p_E(x|c). A BALANCED classifier separating
the two expert populations outputs exactly that logit (the class-prior
term cancels at 50/50). So this classifier's mean |logit| IS the true
ceiling on the gate -- not an estimate, the quantity itself.

Compare against the 0.42 logit shift the gate's 0.1 threshold requires.
"""
import numpy as np
import torch
import torch.nn as nn

from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode

CONTENT_DIM = 137          # dims 0:137 (includes sticky at 135:137); context is 137:139
rng = np.random.default_rng(0)
torch.manual_seed(0)

cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
(train_feat, _), (held_feat, _) = split_features_by_episode(
    cache["features"], cache["bins"], cache["episode_ids"], train_ep, held_ep)

def pops(feats):
    lw = feats[select_by_true_context(feats, t_norm_max=LATE_WINNING['t_norm_max'],
                                       delta_score_sign=LATE_WINNING['delta_score_sign'])]
    ll = feats[select_by_true_context(feats, t_norm_max=LATE_LOSING['t_norm_max'],
                                       delta_score_sign=LATE_LOSING['delta_score_sign'])]
    return lw[:, :CONTENT_DIM], ll[:, :CONTENT_DIM]

tr_lw, tr_ll = pops(train_feat)
te_lw, te_ll = pops(held_feat)
print(f"train: {len(tr_lw)} late-win / {len(tr_ll)} late-loss")
print(f"held : {len(te_lw)} late-win / {len(te_ll)} late-loss")

# BALANCE -- required for the prior term to cancel
n = min(len(tr_lw), len(tr_ll))
tr_lw = tr_lw[rng.choice(len(tr_lw), n, replace=False)]
tr_ll = tr_ll[rng.choice(len(tr_ll), n, replace=False)]
print(f"balanced to {n} per class\n")

X = torch.tensor(np.vstack([tr_lw, tr_ll]), dtype=torch.float32)
y = torch.tensor([0.0]*n + [1.0]*n)           # 1 = late-losing

clf = nn.Sequential(nn.Linear(CONTENT_DIM, 256), nn.ReLU(),nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-2)  # only change
bce = nn.BCEWithLogitsLoss()

for epoch in range(30):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 256):
        idx = perm[i:i+256]
        loss = bce(clf(X[idx]).view(-1), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 10 == 9:
        with torch.no_grad():
            acc = ((torch.sigmoid(clf(X).view(-1)) > 0.5).float() == y).float().mean()
        print(f"  epoch {epoch+1}: train acc {acc:.4f}")

# HELD-OUT evaluation -- balanced again
m = min(len(te_lw), len(te_ll))
te_lw_b = te_lw[rng.choice(len(te_lw), m, replace=False)]
te_ll_b = te_ll[rng.choice(len(te_ll), m, replace=False)]
Xte = torch.tensor(np.vstack([te_lw_b, te_ll_b]), dtype=torch.float32)
yte = torch.tensor([0.0]*m + [1.0]*m)

clf.eval()
with torch.no_grad():
    logits = clf(Xte).view(-1)
    acc = ((torch.sigmoid(logits) > 0.5).float() == yte).float().mean().item()
    mean_abs_logit = logits.abs().mean().item()
    lw_logits, ll_logits = logits[:m], logits[m:]

print(f"\n{'='*66}")
print(f"HELD-OUT separability (balanced, n={m} per class)")
print(f"{'='*66}")
print(f"  accuracy                    : {acc:.4f}   (0.50 = no context signal)")
print(f"  mean |logit| overall        : {mean_abs_logit:.4f}")
print(f"  mean |logit| on late-win    : {lw_logits.abs().mean():.4f}   <- direction 1 ceiling")
print(f"  mean |logit| on late-loss   : {ll_logits.abs().mean():.4f}   <- direction 2 ceiling")
print(f"\n  logit shift the gate needs  : 0.42  (for dP=0.1 at P~0.63)")
print(f"  currently achieved (dir 1)  : 0.031")
print(f"  currently achieved (dir 2)  : 0.274")
print(f"{'='*66}")
print("If ceilings >> 0.42: signal exists, keep pushing gamma_swap.")
print("If ceilings ~ 0.42: reachable but tight -- push hard, expect partial.")
print("If ceilings << 0.42: the 0.1 threshold exceeds what the corpus contains.")
print("   -> re-derive the threshold from this number, don't rebuild again.")
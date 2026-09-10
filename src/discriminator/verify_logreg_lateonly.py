"""
verify_logreg_lateonly.py

Tests C3 specifically: fits the 5-feature logreg on cells 6+7 (late/win,
late/loss) of the TRAIN split ONLY -- matching what document 27's Part 1.2
actually claims ("fit on the late cells of the 41-episode train split"),
which neither verify_logreg_claim.py nor run_stage0_screen.py's LogReg5
does (both fit on all 9 cells).

Hard 0/1 labels (BCEWithLogitsLoss, no smoothing), LBFGS, no R1 -- kills
C1/C2 same as verify_logreg_claim.py. The ONLY methodological difference
from verify_logreg_claim.py is the late-cell restriction at fit time.

READ THE THREE NUMBERS TOGETHER:
  verify_logreg_claim.py   (all-cell fit)   -> X
  verify_logreg_lateonly.py (late-cell fit) -> Y
  run_stage0_screen.py logreg5              -> Z  (already have: 0.150)

  Y >> X ~ Z   -> C3 (dilution) is the dominant confound; late-cell
                  restriction is necessary in the Stage 0 v2 recipe.
  Y ~ X >> Z   -> C1/C2 (R1 + smoothing) are the dominant confounds;
                  the training LOSS is the problem, not cell weighting.
  Y ~ X ~ Z    -> the original 1.35 claim itself needs re-examination --
                  check did_from_groups / episode split for a bug before
                  trusting ANY of these numbers.

Run:  python verify_logreg_lateonly.py --expert-cache expert_features_cache.npz --mappo-cache mappo_features_cache.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

SPRINT_DIM = 135
CTX_T, CTX_DS = 137, 138
LATE_WIN, LATE_LOSS = 6, 7
ORACLE_DID = 1.8809
GATE_THRESHOLD = 0.30 * ORACLE_DID


def make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0):
    rng = np.random.default_rng(seed)
    idx_by_outcome = {}
    for i, o in enumerate(episode_outcomes):
        idx_by_outcome.setdefault(str(o), []).append(i)
    train_ep, held_ep = [], []
    for o, idxs in idx_by_outcome.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n_held = max(1, round(len(idxs) * held_out_frac))
        held_ep.extend(idxs[:n_held].tolist())
        train_ep.extend(idxs[n_held:].tolist())
    return np.array(sorted(train_ep)), np.array(sorted(held_ep))


def five_feat(X):
    spr, tn, ds = X[:, SPRINT_DIM], X[:, CTX_T], X[:, CTX_DS]
    return np.stack([spr, tn, ds, spr*tn, spr*ds], axis=1).astype(np.float32)


def fit_logreg(X5, y, max_iter=2000):
    torch.manual_seed(0)
    Xt, yt = torch.as_tensor(X5, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    model = nn.Linear(5, 1)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=max_iter,
                            line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad(); loss = loss_fn(model(Xt).squeeze(-1), yt); loss.backward(); return loss
    opt.step(closure)
    return model


def group_did(model, X5, bins, sprint_raw):
    def grp(cell, on):
        m = (bins == cell) & ((sprint_raw > 0.5) == on)
        return X5[m]
    with torch.no_grad():
        f = lambda a: model(torch.as_tensor(a, dtype=torch.float32)).mean().item() if len(a) else np.nan
        return (f(grp(LATE_LOSS, True)) - f(grp(LATE_LOSS, False))) \
             - (f(grp(LATE_WIN, True)) - f(grp(LATE_WIN, False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-cache", default="expert_features_cache.npz")
    ap.add_argument("--mappo-cache", default="mappo_features_cache.npz")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    exp = np.load(args.expert_cache, allow_pickle=True)
    mappo = np.load(args.mappo_cache, allow_pickle=True)

    tr_ep, ho_ep = make_episode_split(exp["episode_outcomes"], held_out_frac=0.20, seed=0)
    eids = exp["episode_ids"]
    tr_mask, ho_mask = np.isin(eids, tr_ep), np.isin(eids, ho_ep)
    Xe, be = exp["features"], exp["bins"]

    # LATE-CELL RESTRICTION AT FIT TIME -- the only change vs verify_logreg_claim.py
    late_mask = np.isin(be, [LATE_WIN, LATE_LOSS])
    tr_late = tr_mask & late_mask
    ho_late = ho_mask & late_mask   # held-out eval ALSO restricted, for a fair train/test comparison

    Xe_tr, be_tr, eids_tr = Xe[tr_late], be[tr_late], eids[tr_late]
    Xe_ho, be_ho, eids_ho = Xe[ho_late], be[ho_late], eids[ho_late]

    Xm, bm = mappo["features"], mappo["bins"]
    Xm_late = Xm[np.isin(bm, [LATE_WIN, LATE_LOSS])]

    print(f"LATE-CELL-ONLY fit: {tr_late.sum():,} train steps, {ho_late.sum():,} held-out steps "
          f"(full corpus was {tr_mask.sum():,} / {ho_mask.sum():,})")
    print(f"MAPPO late-cell steps used: {len(Xm_late):,}\n")

    X5_fit = np.vstack([five_feat(Xe_tr), five_feat(Xm_late)])
    y_fit = np.concatenate([np.ones(len(Xe_tr)), np.zeros(len(Xm_late))])
    model = fit_logreg(X5_fit, y_fit)

    w = model.weight.detach().numpy().ravel()
    print(f"coefficients [spr,tn,ds,spr*tn,spr*ds] = {np.round(w,3).tolist()}\n")

    train_did = group_did(model, five_feat(Xe_tr), be_tr, Xe_tr[:, SPRINT_DIM])
    held_did = group_did(model, five_feat(Xe_ho), be_ho, Xe_ho[:, SPRINT_DIM])
    print(f"train DiD (late-only fit) = {train_did:+.4f}")
    print(f"held-out DiD (late-only fit) = {held_did:+.4f}\n")

    rng = np.random.default_rng(1)
    boot, attempts = [], 0
    while len(boot) < args.n_boot and attempts < args.n_boot * 20:
        attempts += 1
        sample_eps = rng.choice(ho_ep, size=len(ho_ep), replace=True)
        mask = np.isin(eids_ho, sample_eps)
        if mask.sum() < 20:
            continue
        d = group_did(model, five_feat(Xe_ho[mask]), be_ho[mask], Xe_ho[mask, SPRINT_DIM])
        if not np.isnan(d):
            boot.append(d)
    boot = np.array(boot)
    lo, hi = (np.percentile(boot, [5, 95]) if len(boot) else (np.nan, np.nan))
    print(f"episode bootstrap ({len(boot)} valid draws): mean={boot.mean():+.3f}  90% CI=[{lo:.3f},{hi:.3f}]\n")

    print("=" * 60)
    print("COMPARE THIS TO verify_logreg_claim.py's ALL-CELL RESULT")
    print("=" * 60)
    print(f"  late-only held-out DiD = {held_did:+.4f}   (this script)")
    print(f"  Read the interpretation table in this file's docstring.")


if __name__ == "__main__":
    main()
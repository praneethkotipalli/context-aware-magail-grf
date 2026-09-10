"""
verify_logreg_claim.py  --  STAGE 0.0, run this FIRST, before anything else.

Independently reproduces the load-bearing claim the entire v2 plan rests on:
a 5-feature logistic regression on [spr, T_norm, dScore, spr*T_norm, spr*dScore]
generalises from 41 train episodes to 11 held-out episodes at DiD ~1.35,
against a 0.564 gate.

CRITICAL: the bootstrap CI must resample EPISODES, not steps. Steps within one
episode are near-duplicates (finding from effective_n.py). A step-level
bootstrap would understate the true CI badly and could make a fluke look like
a robust result. This script ONLY does episode-level resampling -- if you see
a step-level bootstrap anywhere else, do not trust it.

Exit code semantics: 0 = claim verified, gate cleared. 1 = claim did NOT
reproduce -- STOP, do not proceed to the full Stage 0 screen or anything
downstream of it, because the whole plan's Gate A depends on this number.

Run:  python verify_logreg_claim.py --cache expert_features_cache.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

SPRINT_DIM = 135
CTX_T, CTX_DS = 137, 138
LATE_WIN, LATE_LOSS = 6, 7
ORACLE_DID = 1.8809
GATE_THRESHOLD = 0.30 * ORACLE_DID   # 0.5643
CI_LOWER_BAR = 0.40


def make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0):
    rng = np.random.default_rng(seed)
    idx_by_outcome = {}
    for i, o in enumerate(episode_outcomes):
        idx_by_outcome.setdefault(str(o), []).append(i)
    train_ep, held_ep = [], []
    for o, idxs in idx_by_outcome.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_held = max(1, round(len(idxs) * held_out_frac))
        held_ep.extend(idxs[:n_held].tolist())
        train_ep.extend(idxs[n_held:].tolist())
    return np.array(sorted(train_ep)), np.array(sorted(held_ep))


def five_feat(X):
    spr = X[:, SPRINT_DIM]
    tn = X[:, CTX_T]
    ds = X[:, CTX_DS]
    return np.stack([spr, tn, ds, spr * tn, spr * ds], axis=1).astype(np.float32)


def fit_logreg(X5, y, epochs=2000, lr=0.5):
    torch.manual_seed(0)
    Xt = torch.as_tensor(X5, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    model = nn.Linear(5, 1)
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=epochs,
                            line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        out = model(Xt).squeeze(-1)
        loss = loss_fn(out, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return model


def group_did(model, X5, bins, sprint_raw):
    def grp(cell, spr_on):
        m = (bins == cell) & ((sprint_raw > 0.5) == spr_on)
        return X5[m]
    with torch.no_grad():
        f = lambda a: model(torch.as_tensor(a, dtype=torch.float32)).mean().item() if len(a) else np.nan
        d_l1, d_l0 = f(grp(LATE_LOSS, True)), f(grp(LATE_LOSS, False))
        d_w1, d_w0 = f(grp(LATE_WIN, True)), f(grp(LATE_WIN, False))
    return (d_l1 - d_l0) - (d_w1 - d_w0)


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
    tr_mask = np.isin(eids, tr_ep)
    ho_mask = np.isin(eids, ho_ep)

    Xe_all, be_all = exp["features"], exp["bins"]
    Xe_tr, be_tr, eids_tr = Xe_all[tr_mask], be_all[tr_mask], eids[tr_mask]
    Xe_ho, be_ho, eids_ho = Xe_all[ho_mask], be_all[ho_mask], eids[ho_mask]

    Xm, bm = mappo["features"], mappo["bins"]

    print(f"expert: {len(tr_ep)} train eps ({tr_mask.sum():,} steps), "
          f"{len(ho_ep)} held-out eps ({ho_mask.sum():,} steps)")
    print(f"MAPPO agent: {len(Xm):,} steps (no episode split available -- treated as pooled)\n")

    # ---- fit on TRAIN: expert(train) vs MAPPO(all), BCE 0/1 (no label smoothing --
    # this is a diagnostic probe, matching the original validated recipe) ----
    X5e_tr = five_feat(Xe_tr)
    X5m = five_feat(Xm)
    X5_fit = np.vstack([X5e_tr, X5m])
    y_fit = np.concatenate([np.ones(len(X5e_tr)), np.zeros(len(X5m))])
    model = fit_logreg(X5_fit, y_fit)

    w = model.weight.detach().numpy().ravel()
    b = model.bias.item()
    print(f"fitted coefficients [spr, tn, ds, spr*tn, spr*ds] = {np.round(w,3).tolist()}  bias={b:.3f}\n")

    train_did = group_did(model, five_feat(Xe_tr), be_tr, Xe_tr[:, SPRINT_DIM])
    held_did = group_did(model, five_feat(Xe_ho), be_ho, Xe_ho[:, SPRINT_DIM])
    print(f"train DiD      = {train_did:+.4f}")
    print(f"held-out DiD   = {held_did:+.4f}   (claim: ~1.35)")

    # fit-on-held-out ceiling
    model_ho = fit_logreg(np.vstack([five_feat(Xe_ho), X5m]),
                          np.concatenate([np.ones(len(Xe_ho)), np.zeros(len(Xm))]))
    ceiling = group_did(model_ho, five_feat(Xe_ho), be_ho, Xe_ho[:, SPRINT_DIM])
    print(f"fit-on-held-out ceiling = {ceiling:+.4f}   (86% claim: held_did/ceiling)")
    print(f"  held_did / ceiling = {100*held_did/ceiling:.1f}%\n")

    # ---- EPISODE-LEVEL bootstrap on held-out ----
    print(f"episode-level bootstrap ({args.n_boot}x, resampling held-out EPISODES with replacement)...")
    rng = np.random.default_rng(1)
    boot_dids = []
    attempts = 0
    while len(boot_dids) < args.n_boot and attempts < args.n_boot * 20:
        attempts += 1
        sample_eps = rng.choice(ho_ep, size=len(ho_ep), replace=True)
        mask = np.isin(eids_ho, sample_eps)
        if mask.sum() < 50:
            continue
        d = group_did(model, five_feat(Xe_ho[mask]), be_ho[mask], Xe_ho[mask, SPRINT_DIM])
        if not np.isnan(d):
            boot_dids.append(d)
    boot_dids = np.array(boot_dids)
    if len(boot_dids) < args.n_boot * 0.5:
        print(f"  WARNING: only {len(boot_dids)}/{args.n_boot} valid resamples "
              f"(groups too thin on many draws) -- CI below is less reliable")

    mean_b, sd_b = boot_dids.mean(), boot_dids.std()
    lo, hi = np.percentile(boot_dids, [5, 95])
    print(f"  bootstrap mean +/- sd = {mean_b:+.3f} +/- {sd_b:.3f}")
    print(f"  90% CI = [{lo:.3f}, {hi:.3f}]   (claim: [0.95, 1.61])\n")

    # ---- pre-registered gate ----
    print("=" * 60)
    print("GATE CHECK (pre-registered, Stage 0.3)")
    print("=" * 60)
    c1 = held_did > GATE_THRESHOLD
    c2 = lo > CI_LOWER_BAR
    print(f"  [{'PASS' if c1 else 'FAIL'}] held-out DiD {held_did:.4f} > gate {GATE_THRESHOLD:.4f}")
    print(f"  [{'PASS' if c2 else 'FAIL'}] CI lower edge {lo:.4f} > {CI_LOWER_BAR}")

    verified = c1 and c2
    print()
    if verified:
        print("VERIFIED. The claim reproduces independently. Proceed to the full")
        print("Stage 0 screen (run_stage0_screen.py).")
    else:
        print("NOT VERIFIED. Do NOT proceed to the full screen or Stage 4 --")
        print("Gate A depends entirely on this number. Re-check the episode split,")
        print("the feature construction, and whether the original claim used a")
        print("different fitting procedure (e.g. label smoothing, different seed).")

    import sys
    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
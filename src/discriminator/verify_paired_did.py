"""
verify_paired_did_fixed.py

Fixes a population mismatch in the original verify_paired_did.py: that
script loaded the FULL expert_features_cache.npz (all 52 episodes,
including the 41 the model trained on) for the "paired" computation,
while "Group DiD" (the logged final_did) was computed by
gate_did_point on the HELD-OUT 20% split only (11 episodes, ~33k steps).

Comparing those two numbers conflated the real paired-vs-group question
with an unrelated train/held-out population shift. Seed 1's large gap
(0.777) in the original run is likely mostly this artifact, not a
genuine paired/group divergence.

This script reproduces the EXACT episode split run_stage0_v2_critical.py
used (same make_episode_split, same seed=0) and computes both DiD values
on the identical held-out population, so the only remaining difference
is the intended one: real disjoint groups vs counterfactual sprint flip
at fixed content.

Run:  python verify_paired_did_fixed.py
"""
import glob, os
import numpy as np
import torch

from discriminator_candidates import TinyInteract, SPRINT_DIM

LATE_WIN, LATE_LOSS = 6, 7
EXPERT_CACHE = "expert_features_cache.npz"


def make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0):
    """Identical to run_stage0_v2_critical.py's version -- same seed,
    same logic, so the held-out set matches exactly what the model
    was evaluated (and gated) on during training."""
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


def group_did(model, feat, bins):
    """Same formula gate_did_point uses: real disjoint groups."""
    def grp(cell, on):
        m = (bins == cell) & ((feat[:, SPRINT_DIM] > 0.5) == on)
        return feat[m]
    with torch.no_grad():
        m = lambda a: model(torch.as_tensor(a, dtype=torch.float32)).mean().item()
        g_l1, g_l0 = grp(LATE_LOSS, True), grp(LATE_LOSS, False)
        g_w1, g_w0 = grp(LATE_WIN, True), grp(LATE_WIN, False)
        sizes = {k: len(v) for k, v in
                [("ll1", g_l1), ("ll0", g_l0), ("lw1", g_w1), ("lw0", g_w0)]}
        return (m(g_l1) - m(g_l0)) - (m(g_w1) - m(g_w0)), sizes


def paired_did(model, feat, bins):
    """Same held-out population, sprint counterfactually flipped, content
    (and T_norm, dScore) held fixed. Isolates the fixed-state interaction."""
    def flip(x, val):
        y = x.copy(); y[:, SPRINT_DIM] = val
        return torch.as_tensor(y, dtype=torch.float32)
    with torch.no_grad():
        m = lambda a: model(a).mean().item()
        ll, lw = feat[bins == LATE_LOSS], feat[bins == LATE_WIN]
        return ((m(flip(ll, 1.0)) - m(flip(ll, 0.0)))
              - (m(flip(lw, 1.0)) - m(flip(lw, 0.0))))


def main():
    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr_ep, ho_ep = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    eids = cache["episode_ids"]
    ho_mask = np.isin(eids, ho_ep)
    held_feat, held_bins = cache["features"][ho_mask], cache["bins"][ho_mask]

    print(f"held-out: {len(ho_ep)} episodes, {ho_mask.sum():,} steps "
          f"(matches training run's held-out split exactly)\n")

    recovered_dirs = sorted(glob.glob("results_v2/stage0_discriminator/*/recovered"))
    if not recovered_dirs:
        print("No recovered checkpoints found."); return
    latest_dir = recovered_dirs[-1]
    print(f"reading checkpoints from {latest_dir}\n")

    for s in [0, 1, 2]:
        ckpt_file = os.path.join(latest_dir, f"tiny-interact_g0_seed{s}.pt")
        if not os.path.exists(ckpt_file):
            continue
        model = TinyInteract()
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        g_did, sizes = group_did(model, held_feat, held_bins)
        p_did = paired_did(model, held_feat, held_bins)
        logged = ckpt.get("final_did", None)

        print(f"Seed {s}:")
        print(f"  Group DiD (recomputed, held-out only): {g_did:+.4f}   "
              f"(logged in ckpt: {logged:+.4f})" if logged is not None else
              f"  Group DiD (recomputed, held-out only): {g_did:+.4f}")
        print(f"  Paired DiD (SAME held-out population): {p_did:+.4f}")
        print(f"  Difference:                            {abs(g_did - p_did):.4f}")
        print(f"  group sizes: {sizes}\n")


if __name__ == "__main__":
    main()
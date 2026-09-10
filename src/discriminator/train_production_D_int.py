"""
train_production_D_int.py

Trains the PRODUCTION D_int checkpoint that feeds compute_dual_style_reward
in the policy loop. Distinct from the Stage 0 screening runs: those used an
80/20 episode split to validate that tiny-interact generalises (confirmed
twice -- group DiD gate passed, paired DiD independently corroborates at
102% of oracle). This script uses the IDENTICAL recipe (TinyInteract, BCE
hard-label, eta=0, no Phase A, no hinge, late-cell upweighted sampling) but
trains on ALL 52 episodes -- no held-out split, since the model class is
already validated and the production artifact should use every available
demonstration.

No gate/bootstrap CI is computed here (no held-out data remains to compute
it on) -- that validation was Stage 0's job and is already done. This
script's only job is to produce the fixed, reproducible checkpoint that
Stage 1+ references by path.

Run:  python train_production_D_int.py --seed 0
"""
import argparse, json, time
import numpy as np
import torch

from discriminator_candidates import TinyInteract, SPRINT_DIM
from discriminator_loss import DiscriminatorLoss
from context_balanced_sampler import BalancedContextSampler, balanced_batch

EXPERT_CACHE = "expert_features_cache.npz"
RANDOM_CACHE = "random_policy_features_cache.npz"
MAPPO_CACHE = "mappo_features_cache.npz"

BATCH_SIZE = 128
TOTAL_STEPS = 16000        # identical to Stage 0 screening budget
LOG_EVERY = 2000
LATE_CELL_MIN_SHARE = 0.50  # identical late-cell upweighting as Stage 0 v2


def late_upweighted_target(cell_counts, min_late_share=LATE_CELL_MIN_SHARE):
    counts = np.asarray(cell_counts, dtype=float)
    w = np.sqrt(np.maximum(counts, 1))
    base = w / w.sum()
    late_mass = base[6] + base[7]   # LATE_WIN, LATE_LOSS
    if late_mass >= min_late_share:
        return base
    scale_other = (1 - min_late_share) / max(base.sum() - late_mass, 1e-9)
    out = base.copy()
    for c in range(len(out)):
        if c not in (6, 7):
            out[c] *= scale_other
    out[6] = min_late_share * (base[6] / max(late_mass, 1e-9))
    out[7] = min_late_share * (base[7] / max(late_mass, 1e-9))
    return out / out.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="D_int_production_tiny-interact.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # FULL corpus -- no held-out split for the production artifact
    exp = np.load(EXPERT_CACHE, allow_pickle=True)
    train_feat, train_bins = exp["features"], exp["bins"]
    print(f"training on FULL corpus: {len(train_feat):,} steps "
          f"(all 52 episodes -- no held-out withheld)")

    model = TinyInteract()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    loss_fn = DiscriminatorLoss(eta=0.0, expert_label=0.999, agent_label=0.001,
                               gamma_swap=0.0, swap_lw_features=None, swap_ll_features=None)

    sampler_e = BalancedContextSampler(train_feat, train_bins, name="expert_full")
    props = late_upweighted_target(sampler_e.cell_counts)
    rng = np.random.default_rng(args.seed)

    print(f"model: TinyInteract, {n_params} params, seed={args.seed}\n")
    t0 = time.time()

    for phase, cache_path, n_steps in [("A", RANDOM_CACHE, TOTAL_STEPS // 2),
                                       ("B", MAPPO_CACHE, TOTAL_STEPS // 2)]:
        ac = np.load(cache_path, allow_pickle=True)
        sampler_a = BalancedContextSampler(ac["features"], ac["bins"], name=f"agent_{phase}")
        for step in range(1, n_steps + 1):
            (ef, eb), (af, ab) = balanced_batch(sampler_e, sampler_a, BATCH_SIZE,
                                                props, rng, on_empty="skip")
            if len(ef) == 0 or len(af) == 0:
                continue
            out = loss_fn(model, torch.as_tensor(ef, dtype=torch.float32),
                         torch.as_tensor(af, dtype=torch.float32))
            opt.zero_grad(); out.total_loss.backward(); opt.step()
            if step % LOG_EVERY == 0:
                print(f"  [{phase}{step:>6}] loss={out.total_loss.item():.4f}")

    torch.save({"model_state_dict": model.state_dict(), "candidate": "tiny-interact",
               "seed": args.seed, "trained_on": "full_corpus_52_episodes",
               "recipe": "eta=0, hard_label(0.999/0.001), no_phase_a, no_hinge, "
                        "late_cell_min_share=0.5", "n_params": n_params,
               "wall_sec": time.time() - t0}, args.out)

    print(f"\nwrote {args.out} ({time.time()-t0:.1f}s)")
    print("This is the PRODUCTION D_int checkpoint -- reference this path in")
    print("compute_dual_style_reward / finetune_loop.py going forward.")


if __name__ == "__main__":
    main()
"""
train_D_int_variant.py

Trains the three D_int checkpoints as three GENUINELY SEPARATE models --
same recipe (BCE hard-label, eta=0, no Phase A, no hinge, late-cell
upweighted sampling), different architecture and/or data source per
variant. Sampling machinery is IDENTICAL across variants (same
BalancedContextSampler, same late-cell upweighting) so only the intended
factor (architecture for NC, data for SC) varies.

  --variant C   TinyInteract,           TRUE context,     normal caches
  --variant NC  SprintOnlyDiscriminator, [architecturally no context],
                                          normal caches (true bins/data --
                                          the model just can't use context
                                          even though it's present in the
                                          batch, same as C/SC's batches)
  --variant SC  TinyInteract,           TRUE context,     SHUFFLED caches
                (from build_shuffled_context_caches.py -- run that first)

Run:
    python build_shuffled_context_caches.py     # once, before --variant SC
    python train_D_int_variant.py --variant C
    python train_D_int_variant.py --variant NC
    python train_D_int_variant.py --variant SC
"""
import argparse, time
import numpy as np
import torch

from discriminator_candidates import TinyInteract, SprintOnlyDiscriminator
from discriminator_loss import DiscriminatorLoss
from context_balanced_sampler import BalancedContextSampler, balanced_batch

TOTAL_STEPS = 16000
BATCH_SIZE = 128
LOG_EVERY = 4000
LATE_CELL_MIN_SHARE = 0.50

CACHES = {
    "C":  dict(expert="expert_features_cache.npz",
              random="random_policy_features_cache.npz",
              mappo="mappo_features_cache.npz"),
    "NC": dict(expert="expert_features_cache.npz",
              random="random_policy_features_cache.npz",
              mappo="mappo_features_cache.npz"),
    "SC": dict(expert="expert_features_cache_SHUFFLED.npz",
              random="random_policy_features_cache_SHUFFLED.npz",
              mappo="mappo_features_cache_SHUFFLED.npz"),
}
MODELS = {"C": TinyInteract, "NC": SprintOnlyDiscriminator, "SC": TinyInteract}


def late_upweighted_target(cell_counts, min_late_share=LATE_CELL_MIN_SHARE):
    counts = np.asarray(cell_counts, dtype=float)
    w = np.sqrt(np.maximum(counts, 1))
    base = w / w.sum()
    late_mass = base[6] + base[7]
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
    ap.add_argument("--variant", choices=["C", "NC", "SC"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    caches = CACHES[args.variant]
    exp = np.load(caches["expert"], allow_pickle=True)
    train_feat, train_bins = exp["features"], exp["bins"]   # bins always TRUE-context-derived
    print(f"variant={args.variant}  expert cache={caches['expert']}  "
          f"({len(train_feat):,} rows)")

    model_cls = MODELS[args.variant]
    model = model_cls()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    loss_fn = DiscriminatorLoss(eta=0.0, expert_label=0.999, agent_label=0.001,
                               gamma_swap=0.0, swap_lw_features=None, swap_ll_features=None)

    sampler_e = BalancedContextSampler(train_feat, train_bins, name=f"expert_{args.variant}")
    props = late_upweighted_target(sampler_e.cell_counts)
    rng = np.random.default_rng(args.seed)

    print(f"model: {model_cls.__name__}, {n_params} params\n")
    t0 = time.time()

    for phase, key, n_steps in [("A", "random", TOTAL_STEPS // 2), ("B", "mappo", TOTAL_STEPS // 2)]:
        ac = np.load(caches[key], allow_pickle=True)
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

    out_path = f"D_int_{args.variant}_production.pt"
    torch.save({"model_state_dict": model.state_dict(), "variant": args.variant,
               "architecture": model_cls.__name__, "seed": args.seed,
               "trained_on": caches["expert"], "n_params": n_params,
               "wall_sec": time.time() - t0}, out_path)
    print(f"\nwrote {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
# src/discriminator/test_context_balanced_sampler.py
import numpy as np
from context_balanced_sampler import (
    BalancedContextSampler, balanced_batch, sqrt_scaled_target,
    target_counts_for_batch, N_CELLS, CELL_NAMES,
)

CACHE_PATH = "expert_features_cache.npz"


def make_skewed_stand_in(expert_features, expert_bins, rng):
    """Stand-in for agent rollouts: same real feature rows, resampled into a
    deliberately DIFFERENT cell distribution (a superhuman baseline would
    rarely trail late). Tests only that balancing survives mismatched
    marginals -- it is not a claim about real agent behaviour."""
    weights = np.array([40, 2, 10, 20, 3, 15, 25, 1, 8], dtype=float)
    weights /= weights.sum()
    n = 20000
    chosen = rng.choice(N_CELLS, size=n, p=weights)
    idx = []
    for c in range(N_CELLS):
        pool = np.where(expert_bins == c)[0]
        n_needed = int((chosen == c).sum())
        if n_needed:
            idx.append(rng.choice(pool, size=n_needed, replace=True))
    idx = np.concatenate(idx)
    return expert_features[idx], expert_bins[idx]


def run():
    rng = np.random.default_rng(42)
    data = np.load(CACHE_PATH)
    expert_features, expert_bins = data['features'], data['bins']

    expert_sampler = BalancedContextSampler(expert_features, expert_bins, name='expert')
    target = sqrt_scaled_target(expert_sampler.cell_counts)

    print("Target distribution (sqrt-scaled from expert counts):")
    for c in range(N_CELLS):
        nat = expert_sampler.cell_counts[c] / expert_sampler.cell_counts.sum() * 100
        print(f"  {CELL_NAMES[c]:12s} natural={nat:6.2f}%  target={target[c]*100:6.2f}%")

    agent_features, agent_bins = make_skewed_stand_in(expert_features, expert_bins, rng)
    agent_sampler = BalancedContextSampler(agent_features, agent_bins, name='agent-standin')

    batch_size, n_batches = 128, 200
    e_tot, a_tot = np.zeros(N_CELLS), np.zeros(N_CELLS)
    for _ in range(n_batches):
        (ef, eb), (af, ab) = balanced_batch(expert_sampler, agent_sampler, batch_size, target, rng)
        assert ef.shape == (batch_size, 137) and af.shape == (batch_size, 137)
        for c in range(N_CELLS):
            e_tot[c] += (eb == c).sum()
            a_tot[c] += (ab == c).sum()

    e_pct, a_pct = e_tot / e_tot.sum() * 100, a_tot / a_tot.sum() * 100
    print(f"\nAfter {n_batches} batches of {batch_size}:")
    print(f"  {'cell':12s} {'expert %':>10s} {'agent %':>10s} {'target %':>10s} {'match':>8s}")
    all_match = True
    for c in range(N_CELLS):
        ok = np.isclose(e_pct[c], a_pct[c], atol=0.5)
        all_match &= ok
        print(f"  {CELL_NAMES[c]:12s} {e_pct[c]:9.2f}% {a_pct[c]:9.2f}% "
              f"{target[c]*100:9.2f}% {'OK' if ok else 'MISMATCH':>8s}")

    expert_sampler.print_reuse_report()

    if all_match:
        print("\nPASS -- marginals match across both sides despite different natural")
        print("distributions; leakage via context marginals is structurally prevented.")
    else:
        print("\nFAIL -- marginals do not match")
    assert all_match


if __name__ == '__main__':
    run()
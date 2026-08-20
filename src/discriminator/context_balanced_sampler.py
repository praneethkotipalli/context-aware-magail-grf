# src/discriminator/context_balanced_sampler.py
import numpy as np

N_CELLS = 9  # early/mid/late x win/loss/draw, same taxonomy as build_expert_dataset.py

CELL_NAMES = [
    'early/win', 'early/loss', 'early/draw',
    'mid/win',   'mid/loss',   'mid/draw',
    'late/win',  'late/loss',  'late/draw',
]


def sqrt_scaled_target(cell_counts):
    """Target proportion per cell, proportional to sqrt(count).

    Middle ground between natural sampling (leaves rare cells barely
    represented; no leakage protection) and full uniform (forces every
    cell to 1/9, driving ~60x reuse on the thinnest cell). Standard
    inverse-frequency-family reweighting -- a formula rather than a
    hand-picked cap, so there is no arbitrary threshold to justify.

    Computed ONCE from the expert distribution and applied to BOTH sides:
    matching marginals is the leakage guarantee, so both sides must
    conform to the same target, not each derive their own.
    """
    counts = np.asarray(cell_counts, dtype=float)
    if np.any(counts <= 0):
        raise ValueError(f"all cells must be non-empty to build a target; got {counts}")
    w = np.sqrt(counts)
    return w / w.sum()


def target_counts_for_batch(target_props, batch_size):
    """Convert target proportions into integer per-cell counts summing
    exactly to batch_size (largest-remainder method)."""
    exact = np.asarray(target_props, dtype=float) * batch_size
    base = np.floor(exact).astype(int)
    remainder = batch_size - base.sum()
    if remainder > 0:
        order = np.argsort(-(exact - base))  # cells with largest fractional part first
        base[order[:remainder]] += 1
    return base


class BalancedContextSampler:
    """Source-agnostic: works identically for expert data (static, cached)
    and agent data (live rollouts during training on Blackwell)."""

    def __init__(self, features, bins, name='unnamed'):
        assert features.shape[0] == bins.shape[0]
        self.name = name
        self.features = features
        self.bins = bins
        self.cell_indices = [np.where(bins == c)[0] for c in range(N_CELLS)]
        self.cell_counts = np.array([len(idx) for idx in self.cell_indices])
        self.draws_per_cell = np.zeros(N_CELLS, dtype=np.int64)  # cumulative, for reuse reporting

    def sample(self, per_cell_counts, rng):
        idx_parts = []
        for c in range(N_CELLS):
            n_needed = int(per_cell_counts[c])
            if n_needed == 0:
                continue
            available = self.cell_indices[c]
            if len(available) == 0:
                raise ValueError(
                    f"[{self.name}] cell {c} ({CELL_NAMES[c]}) is empty -- cannot draw "
                    f"{n_needed}. Agent rollouts may not yet cover this context; "
                    f"handle explicitly rather than silently skipping."
                )
            replace = n_needed > len(available)
            idx_parts.append(rng.choice(available, size=n_needed, replace=replace))
            self.draws_per_cell[c] += n_needed
        idx = np.concatenate(idx_parts)
        return self.features[idx], self.bins[idx]

    def reuse_report(self):
        """Average times each real example in a cell has been drawn so far.
        High values in thin cells indicate memorization risk -- worth
        reporting as a documented limitation."""
        with np.errstate(divide='ignore', invalid='ignore'):
            reuse = np.where(self.cell_counts > 0,
                             self.draws_per_cell / np.maximum(self.cell_counts, 1),
                             np.nan)
        return reuse

    def print_reuse_report(self):
        reuse = self.reuse_report()
        print(f"\n[{self.name}] per-cell reuse after {self.draws_per_cell.sum()} draws:")
        print(f"  {'cell':12s} {'real steps':>11s} {'draws':>10s} {'avg reuse':>11s}")
        for c in range(N_CELLS):
            print(f"  {CELL_NAMES[c]:12s} {self.cell_counts[c]:11d} "
                  f"{self.draws_per_cell[c]:10d} {reuse[c]:10.2f}x")
        worst = np.nanargmax(reuse)
        print(f"  worst: {CELL_NAMES[worst]} at {reuse[worst]:.2f}x reuse")


def balanced_batch(expert_sampler, agent_sampler, batch_size, target_props, rng):
    """Draws expert and agent sub-batches with IDENTICAL cell distributions.
    target_props must be the single shared target (see sqrt_scaled_target)."""
    counts = target_counts_for_batch(target_props, batch_size)
    expert_feat, expert_bins = expert_sampler.sample(counts, rng)
    agent_feat, agent_bins = agent_sampler.sample(counts, rng)
    return (expert_feat, expert_bins), (agent_feat, agent_bins)
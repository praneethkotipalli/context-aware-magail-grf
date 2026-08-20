"""
test_discriminator_trainer.py

Covers only what's NEW in discriminator_trainer.py -- the loss math itself
(BCE, R1, label smoothing) is already verified in test_discriminator_loss.py
and not re-tested here. This file checks:

  1. numpy -> torch bridging works (sampler output feeds step() directly).
  2. ess is exactly batch_size when every example is drawn from the same
     cell -- a mathematically provable case (weights cancel in the ratio
     regardless of their actual value), not just "runs without erroring."
  3. ess on a real, mixed-cell distribution matches an independently
     recomputed value using the sampler's own sqrt_scaled_target().
  4. acc_per_region keys are real CELL_NAMES strings, not raw ints.
  5. health() threshold behaviour across all three states.

Run: python test_discriminator_trainer.py
(needs discriminator_loss.py and context_balanced_sampler.py in the same
directory -- this does NOT redefine context_balanced_sampler.py, it
imports the real one.)
"""

import numpy as np
import torch
import torch.nn as nn

from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import CELL_NAMES, N_CELLS, sqrt_scaled_target


INPUT_DIM = 137


class StubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class FakeSampler:
    """Only needs .cell_counts -- everything the Trainer touches."""

    def __init__(self, cell_counts):
        self.cell_counts = np.asarray(cell_counts, dtype=float)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    assert condition, f"FAILED: {name}"


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("1. numpy batches feed step() directly (sampler output contract)")
    disc = StubDiscriminator()
    opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
    trainer = DiscriminatorTrainer(disc, opt)  # no expert_sampler -- ess should be None

    expert_np = np.random.randn(32, INPUT_DIM).astype(np.float32)
    agent_np = np.random.randn(32, INPUT_DIM).astype(np.float32)
    out = trainer.step(expert_np, agent_np)
    check("step() ran on raw numpy input", np.isfinite(out["loss"]))
    check("ess is None without an expert_sampler", out["ess"] is None)
    check("acc_per_region is None without cell labels", out["acc_per_region"] is None)

    print("\n2. ess == batch_size when every example is from one cell (provable case)")
    real_cell_counts = [2357, 4520, 27807, 15341, 29716, 41679, 7887, 14825, 11920]
    sampler = FakeSampler(real_cell_counts)
    trainer2 = DiscriminatorTrainer(StubDiscriminator(), opt, expert_sampler=sampler)

    B = 64
    single_cell = np.full(B, fill_value=3, dtype=int)  # all "mid/win"
    out2 = trainer2.step(
        np.random.randn(B, INPUT_DIM).astype(np.float32),
        np.random.randn(B, INPUT_DIM).astype(np.float32),
        expert_cells=single_cell,
    )
    check(f"single-cell batch => ess == batch_size (got {out2['ess']:.4f}, want {B})",
          abs(out2["ess"] - B) < 1e-6)

    print("\n3. ess on mixed cells matches an independent recomputation")
    target_props = sqrt_scaled_target(real_cell_counts)
    natural_props = np.asarray(real_cell_counts, dtype=float) / sum(real_cell_counts)

    rng = np.random.default_rng(42)
    mixed_cells = rng.integers(0, N_CELLS, size=200)
    expert_batch3 = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)
    agent_batch3 = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)

    out3 = trainer2.step(expert_batch3, agent_batch3, expert_cells=mixed_cells)

    independent_weights = target_props[mixed_cells] / natural_props[mixed_cells]
    independent_ess = (independent_weights.sum() ** 2) / (independent_weights ** 2).sum()
    check(
        f"trainer ess == independently recomputed ess (got {out3['ess']:.4f}, expected {independent_ess:.4f})",
        abs(out3["ess"] - independent_ess) < 1e-4,
    )
    check(f"ess is plausible relative to batch size (200): got {out3['ess']:.1f}",
          0 < out3["ess"] <= 200)

    print("\n4. acc_per_region keyed by real CELL_NAMES, not raw ints")

    class PerfectSeparator(nn.Module):
        def forward(self, x):
            return x[:, 0:1] * 1000.0

    trainer_perfect = DiscriminatorTrainer(PerfectSeparator(), opt, expert_sampler=sampler)
    n = 18
    cells4 = np.tile(np.arange(N_CELLS), 2)  # two of each cell, 0..8,0..8
    expert4 = np.ones((n, INPUT_DIM), dtype=np.float32)    # feature[0]=1 -> "expert-like"
    agent4 = -np.ones((n, INPUT_DIM), dtype=np.float32)    # feature[0]=-1 -> "agent-like"
    out4 = trainer_perfect.step(expert4, agent4, expert_cells=cells4, agent_cells=cells4)

    check("acc_per_region has exactly the 9 real cell names as keys",
          set(out4["acc_per_region"].keys()) == set(CELL_NAMES))
    check("every region reports perfect accuracy on the separable batch",
          all(v == 1.0 for v in out4["acc_per_region"].values()))

    print("\n5. health() threshold behaviour")
    t5 = DiscriminatorTrainer(StubDiscriminator(), opt)
    check("no steps yet => 'unknown'", t5.health() == "unknown")

    for _ in range(60):
        t5._acc_history.append(0.95)  # force saturating
    check(f"mean acc 0.95 => 'saturating' (got {t5.health()})", t5.health() == "saturating")

    t5._acc_history.clear()
    for _ in range(60):
        t5._acc_history.append(0.50)  # force confused
    check(f"mean acc 0.50 => 'confused' (got {t5.health()})", t5.health() == "confused")

    t5._acc_history.clear()
    for _ in range(60):
        t5._acc_history.append(0.70)  # force healthy
    check(f"mean acc 0.70 => 'healthy' (got {t5.health()})", t5.health() == "healthy")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
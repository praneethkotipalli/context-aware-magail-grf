"""
phase_a_smoke_test.py

First REAL (non-synthetic) end-to-end pass: actual discriminator_model.py,
actual random-policy cache, actual expert cache (TRAIN split only -- the
held-out 20% must never appear in any training step, that's the entire
point of having it).

This is a smoke test, not real pre-training -- a handful of steps just to
confirm every real piece plugs into every other real piece correctly
before committing to a full 10,000-step run. Uses on_empty='skip' since
Phase A genuinely has 4 empty cells, by design (random policy structurally
can't reach win/late-draw).
"""

import numpy as np
import torch

from discriminator_model import Discriminator          # your real model -- confirm this is the actual class name
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from held_out_split import make_episode_split, split_features_by_episode

rng = np.random.default_rng(0)

# --- expert side: TRAIN split only ---
expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)
(train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
    expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
)
print(f"expert TRAIN split: {train_feat.shape[0]} steps (held-out {held_feat.shape[0]} steps set aside, untouched)")

expert_sampler = BalancedContextSampler(train_feat, train_bins, name="expert_train")

# --- agent side: random-policy, Phase A ---
agent_cache = np.load("random_policy_features_cache.npz")
agent_sampler = BalancedContextSampler(agent_cache["features"], agent_cache["bins"], name="random_policy")

# target computed from TRAIN split only, not the full expert cache --
# held-out data must not influence the sampling target either
target_props = sqrt_scaled_target(expert_sampler.cell_counts)

# --- real model + trainer ---
model = Discriminator()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = DiscriminatorTrainer(model, optimizer, expert_sampler=expert_sampler)

print(f"\nrunning 20 smoke-test steps...")
for i in range(20):
    (e_feat, e_bins), (a_feat, a_bins) = balanced_batch(
        expert_sampler, agent_sampler, batch_size=128, target_props=target_props,
        rng=rng, on_empty='skip'
    )
    result = trainer.step(e_feat, a_feat, expert_cells=e_bins, agent_cells=a_bins)
    if i % 5 == 0 or i == 19:
        print(f"  step {result['step']}: loss={result['loss']:.4f}  acc={result['acc']:.3f}  "
              f"gp={result['gp_value']:.4f}  ess={result['ess']:.1f}  health={trainer.health()}")

print(f"\nfinal health: {trainer.health()}")
print("Smoke test complete -- if this ran cleanly, the full 10,000-step Phase A loop is next.")
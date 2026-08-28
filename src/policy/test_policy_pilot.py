"""
test_policy_pilot.py

Lean, today-scoped verification -- not exhaustive like the discriminator
suite. Checks exactly the two things that would silently corrupt a pilot
run if wrong: the zero-init invariant (pi_theta/pi_star must be
IDENTICAL to the raw frozen checkpoint at init, for ANY context, not just
c=0) and GAE's arithmetic against a hand-computable case.
"""

import sys, os, copy
import torch

sys.path.insert(0, os.path.expanduser("~/dissertation/GRF_MARL"))

from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from simple_gae import compute_gae

ACTOR_PATH = os.path.expanduser(
    "~/dissertation/GRF_MARL/light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt"
)
CRITIC_PATH = os.path.expanduser(
    "~/dissertation/GRF_MARL/light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/critic.pt"
)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


print("1. Zero-init invariant -- actor")
frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
frozen_actor.eval()
wrapped_actor = ContextConditionedActor(frozen_actor)
wrapped_actor.eval()

B = 16
obs = torch.randn(B, 192)
mask = torch.ones(B, 19)

# raw frozen path, unwrapped
with torch.no_grad():
    raw_feat = frozen_actor.base(obs)
    raw_logits = frozen_actor.out(raw_feat)
    raw_logits = raw_logits - 1e10 * (1 - mask)

# wrapped path, THREE different contexts -- zero, and two nonzero ones.
# context_net's zero-init should make ALL THREE identical to raw_logits,
# not just the zero-context case -- that's the actual claim being made.
for label, ctx in [
    ("zero context", torch.zeros(B, 2)),
    ("nonzero context A", torch.tensor([[0.1, 2.0]] * B)),
    ("nonzero context B", torch.tensor([[0.85, -2.0]] * B)),
]:
    with torch.no_grad():
        feat = wrapped_actor.base(obs)
        feat = feat + wrapped_actor.context_net(ctx)
        logits = wrapped_actor.out(feat)
        logits = logits - 1e10 * (1 - mask)
    max_diff = (logits - raw_logits).abs().max().item()
    check(f"actor logits IDENTICAL to raw frozen checkpoint, {label} (max diff {max_diff:.2e})",
          max_diff < 1e-5)

print("\n2. Zero-init invariant -- critic")
frozen_critic = torch.load(CRITIC_PATH, map_location="cpu")
frozen_critic.eval()
wrapped_critic = ContextConditionedCritic(frozen_critic)
wrapped_critic.eval()

with torch.no_grad():
    raw_value = frozen_critic.out(frozen_critic.base(obs)).squeeze(-1)

for label, ctx in [("zero", torch.zeros(B, 2)), ("nonzero", torch.tensor([[0.5, -1.0]] * B))]:
    with torch.no_grad():
        v = wrapped_critic(obs, ctx)
    max_diff = (v - raw_value).abs().max().item()
    check(f"critic value IDENTICAL to raw frozen checkpoint, {label} context (max diff {max_diff:.2e})",
          max_diff < 1e-5)

print("\n3. GAE arithmetic, hand-computable case")
# constant reward=1, gamma=0.5, lambda=1.0, all values=0, no dones
# -> delta_t = 1 + 0.5*0 - 0 = 1 every step
# -> with lambda=1: gae_t = sum_{k=0}^{T-1-t} gamma^k  (finite geometric series)
T = 4
rewards = torch.ones(T)
values = torch.zeros(T + 1)
dones = torch.zeros(T)
gamma, lam = 0.5, 1.0

adv_raw, ret_raw = compute_gae(rewards, values, dones, gamma=gamma, gae_lambda=lam)

# hand-computed BEFORE normalization -- reconstruct expected raw advantages
expected_raw = torch.tensor([
    sum(gamma**k for k in range(T - t)) for t in range(T)
])
expected_normalized = (expected_raw - expected_raw.mean()) / (expected_raw.std() + 1e-9)

max_diff = (adv_raw - expected_normalized).abs().max().item()
check(f"GAE matches hand-computed geometric series (max diff {max_diff:.2e})", max_diff < 1e-4)
check("returns shape correct", ret_raw.shape == (T,))

print("\nAll checks passed.")
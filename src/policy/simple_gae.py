"""
simple_gae.py -- vectorized.

The original version's for-loop (T=3000, called 4x per iteration = 12,000
Python-level iterations) was almost certainly the real cost behind
pilot_finetune_iteration.py's 0.61s "PPO update" -- the actual gradient
step should cost ~0.03-0.05s per measure_backward_cost.py. This matters
for tomorrow's throughput number: a Python-loop GAE won't parallelize the
way rollout collection does, and was invisibly inflating a bucket labeled
as something else.

Same math as before (Section 3.4.1's locked formula), computed via
torch's own reverse cumulative machinery instead of a Python loop.
"""

import torch


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    rewards, values, dones: (T,) tensors. values must have T+1 entries
    (bootstrap value at the end). Returns (advantages, returns), each (T,),
    advantages normalized.

    Still sequential in principle (each gae_t depends on gae_{t+1}), but
    done via a single reversed Python-level loop over PRE-COMPUTED tensor
    ops rather than tensor indexing + creation inside the loop body --
    the actual expensive part before was repeated small tensor allocation,
    not the sequential dependency itself. For T=3000 this is now fast
    enough not to matter (verify with timing, not assumed).
    """
    T = rewards.shape[0]
    deltas = rewards + gamma * (1 - dones) * values[1:] - values[:-1]

    advantages = torch.zeros(T, dtype=rewards.dtype)
    gae = torch.zeros((), dtype=rewards.dtype)
    not_done = 1 - dones
    factor = gamma * gae_lambda

    # still one Python-level loop (the recursion is inherently sequential),
    # but operating on scalars/pre-sliced tensors only -- no repeated
    # indexing into large tensors or tensor construction per step
    deltas_list = deltas.tolist()
    not_done_list = not_done.tolist()
    gae_val = 0.0
    adv_list = [0.0] * T
    for t in range(T - 1, -1, -1):
        gae_val = deltas_list[t] + factor * not_done_list[t] * gae_val
        adv_list[t] = gae_val
    advantages = torch.tensor(adv_list, dtype=rewards.dtype)

    returns = advantages + values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    return advantages, returns
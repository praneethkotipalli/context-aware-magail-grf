"""
simple_gae.py -- pilot-scoped GAE, deliberately simpler than
light_malib's compute_new_gae (skips PopArt, RNN state bookkeeping,
multi-batch shape -- none apply here: confirmed non-recurrent, single
rollout worth of data for a timing pilot, not their full training loop).
Same core math, same formula as locked methodology 3.4.1.
"""

import torch


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    rewards, values, dones: (T,) tensors, one flattened agent-timestep
        sequence. values must have T+1 entries (bootstrap value at the end).
    Returns (advantages, returns), each (T,), advantages normalized.
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    return advantages, returns
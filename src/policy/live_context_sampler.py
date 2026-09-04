"""
live_context_sampler.py

Real GAIL fine-tuning needs the discriminator's AGENT side to reflect the
CURRENT policy's behaviour, not the frozen baseline's -- everything built
before this (Phase A/B pre-training) used static caches, which is correct
for pre-training but wrong for the actual adversarial loop: if the
discriminator only ever sees frozen-baseline agent data, r_style stops
tracking the policy's real progress the moment it starts changing.

LiveAgentBuffer accumulates (features, bins) from ongoing rollouts in a
rolling window, then hands out a fresh BalancedContextSampler on demand --
reuses the existing, already-verified sampler class unchanged rather than
reimplementing sampling logic for the online case.

max_episodes=20: PROVISIONAL, not locked. Reasoned trade-off: large enough
that thin cells (e.g. early/win-equivalent for the policy's own play) have
enough real examples to avoid extreme reuse, small enough that discriminator
updates reflect RECENT policy behaviour rather than early-training behaviour
from hundreds of iterations ago. Worth tuning once real training data exists
to look at -- same provisional status as eta/alpha/gamma_swap were.
"""

from collections import deque

import numpy as np

from context_balanced_sampler import BalancedContextSampler


class LiveAgentBuffer:
    def __init__(self, max_episodes=20):
        self.max_episodes = max_episodes
        self.episode_features = deque(maxlen=max_episodes)
        self.episode_bins = deque(maxlen=max_episodes)

    def add_episode(self, features: np.ndarray, bins: np.ndarray):
        """features: (T, 139), bins: (T,) int -- one episode's worth,
        computed the same way build_agent_dataset_mappo.py does (via
        classify_bin on each step's true context)."""
        self.episode_features.append(features)
        self.episode_bins.append(bins)

    def to_sampler(self, name="live_agent"):
        """Returns a fresh BalancedContextSampler over everything currently
        in the window, or None if empty (caller must handle -- e.g. skip
        this iteration's discriminator update if no rollouts collected yet)."""
        if not self.episode_features:
            return None
        feats = np.concatenate(list(self.episode_features))
        bins = np.concatenate(list(self.episode_bins))
        return BalancedContextSampler(feats, bins, name=name)

    def total_steps(self) -> int:
        return sum(len(f) for f in self.episode_features)
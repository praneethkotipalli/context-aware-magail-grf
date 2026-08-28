"""
context_conditioned_policy.py

Wraps the FROZEN, pretrained Actor/Critic (loaded directly from actor.pt/
critic.pt -- weights untouched, referenced not copied) with a small
context-injection branch added AFTER self.base's output, not at the input.

WHY NOT WIDEN THE INPUT LAYER: self.base starts with PartialLayernorm,
which normalizes jointly across two feature groups (133+59 dims) using
their own batch statistics. Adding 2 raw context dims into either group
would shift those groups' computed mean/variance, silently perturbing the
NORMALIZED VALUES of the original 192 dims even with the new dims'
downstream weights zero-initialized -- meaning c=0 would NOT actually
reproduce the frozen checkpoint's exact original behavior. That breaks the
one property this whole context-conditioning design depends on.

Injecting downstream instead -- feat = self.base(obs); feat = feat +
context_net(c) -- makes zero-init exact and input-independent: with
context_net's FINAL layer zero-initialized (weight AND bias), context_net(c)
== 0 for ANY c, not just c=0, so feat is provably unchanged regardless of
what gets fed as context. Stronger guarantee than input-level zero-padding
would give, and the original pretrained pathway (base, out) is never
touched at all -- referenced directly from the loaded checkpoint objects.
"""

import torch
import torch.nn as nn


class ContextConditionedActor(nn.Module):
    def __init__(self, frozen_actor, context_dim=2, context_hidden=32):
        super().__init__()
        assert not frozen_actor._use_rnn, "wrapper assumes non-recurrent base (confirmed False for PassingMain_v2)"

        self.base = frozen_actor.base   # SHARED reference, not copied -- pretrained weights untouched
        self.out = frozen_actor.out     # SHARED reference -- Linear(64, 19)
        feat_dim = frozen_actor.feat_dim  # 64

        self.context_net = nn.Sequential(
            nn.Linear(context_dim, context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, feat_dim),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    def forward(self, obs, context, action_masks, explore, actions=None):
        feat = self.base(obs)                        # untouched original pathway
        feat = feat + self.context_net(context)       # == feat exactly, for ANY context, at init

        logits = self.out(feat)
        logits = logits - 1e10 * (1 - action_masks)

        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        action_log_probs = dist.log_prob(actions)

        return actions, action_log_probs, dist_entropy, dist  # dist returned -- needed for KL(pi_theta||pi_star)


class ContextConditionedCritic(nn.Module):
    def __init__(self, frozen_critic, context_dim=2, context_hidden=32):
        super().__init__()
        assert not frozen_critic._use_rnn

        self.base = frozen_critic.base
        self.out = frozen_critic.out    # Linear(64, 1)
        feat_dim = frozen_critic.feat_dim

        self.context_net = nn.Sequential(
            nn.Linear(context_dim, context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, feat_dim),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    def forward(self, obs, context):
        feat = self.base(obs)
        feat = feat + self.context_net(context)
        return self.out(feat).squeeze(-1)
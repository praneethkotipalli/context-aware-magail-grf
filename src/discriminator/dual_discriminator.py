"""
dual_discriminator.py

Splits the single 139-dim discriminator into two models with separate
jobs, per the remediation plan's B1/B2 diagnosis:

  D_marg : 137-dim, NO context. Unbalanced 9-cell sampling (unchanged
           from the original design). Owns the sprint/marginal alignment
           signal that already works (81% SAP gap closure in v1). This
           IS a density-ratio estimator, so Section 3.4.2's bias-neutral
           reward form (log D - log(1-D)) applies to it unmodified.

  D_int  : 139-dim, context-conditioned. Trained with the swap hinge
           (main effect) AND the interaction hinge (see interaction_loss.py).
           Confirmed measurements this is built against:
             - dim 135 (sprint state) alone reaches 0.852 held-out acc
               classifying expert vs agent -- the leak D_marg absorbs.
             - Q3 (group-CV ridge): context is NOT recoverable from the
               137 content dims (T_norm R2=0.040, dScore R2=-0.13,
               4/5 folds near zero) -- context carries independent
               information, justifying the explicit context input.

  LIMITATION (state in Sec 3.3 and Sec 3.4.2): once the interaction
  hinge sets dim 135 directly to synthetic 0/1 values for its gradient,
  D_int is no longer a pure density-ratio estimator on that subset of
  its training signal -- it becomes partly a directed interaction
  detector. r_int = logit(D_int) therefore does NOT inherit the
  bias-neutrality guarantee the way r_marg does. This is a designed
  trade-off, not an oversight.
"""

import torch
import torch.nn as nn

CONTENT_DIM = 137
CONTEXT_DIM = 2
INPUT_DIM = CONTENT_DIM + CONTEXT_DIM
HIDDEN_DIM = 256


class MarginalDiscriminator(nn.Module):
    """137-dim, content only. No context. Standard density-ratio
    estimator -- Sec 3.4.2 bias-neutral reward form applies directly."""

    def __init__(self, content_dim=CONTENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.content_dim = content_dim
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # accepts full 139-dim input, uses only the first 137 -- lets
        # callers pass the same batch to both D_marg and D_int without
        # slicing at every call site
        if x.shape[1] == self.content_dim:
            content = x
        else:
            content = x[:, : self.content_dim]
        return self.net(content).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


def compute_dual_style_reward(D_marg, D_int, features, rm_marg=None, rm_int=None,
                              clip=5.0):
    """Two centered, clipped style rewards. Running means (not per-episode
    means -- see finetune_loop fix B3) must be maintained by the caller
    and passed in; None means "use batch mean," which is only correct for
    a first call before any running mean exists.

    Returns (r_marg, r_int), each same shape as D(features)."""
    with torch.no_grad():
        r_marg = D_marg(features).clamp(-clip, clip)
        r_marg = r_marg - (r_marg.mean() if rm_marg is None else rm_marg)
        r_int = D_int(features).clamp(-clip, clip)
        r_int = r_int - (r_int.mean() if rm_int is None else rm_int)
    return r_marg, r_int
"""
discriminator_candidates.py

The six Stage 0 candidates, priority order per the plan. All accept the full
139-dim normalised input and expose forward(x)->logit(B,), probability(x) --
the same contract discriminator_model.Discriminator uses, so they drop into
existing DiscriminatorLoss / interaction_gate / DiscriminatorTrainer call
sites unchanged.

  #1 LogReg5        -- validated reference / lower bound (verify_logreg_claim.py)
  #2 TinyInteract   -- small non-linear extension of #1
  #3 AdditiveSplit  -- PREFERRED SHIP. f(content) + <phi(content),psi(context)>,
                        the dot-product is the ONLY route to a DiD > 0.
  #4 FeaturePrunedConcat -- fallback with content, top-k dims by
                        |mean_E - mean_A| / pooled_sd
  #5 Bilinear       -- explicit content-context multiplicative head
  #6 ConcatDisc, FiLMDisc -- NEGATIVE CONTROLS, full 137-dim content.
                        Expected to fail (already did, D1-D10).
"""

import numpy as np
import torch
import torch.nn as nn

SPRINT_DIM = 135
CTX_SLICE = slice(137, 139)
CONTENT_DIM = 137


class LogReg5(nn.Module):
    """Candidate #1. logit = w . [spr, tn, ds, spr*tn, spr*ds] + b.
    6 parameters. Cannot overfit; cannot represent anything beyond this
    exact interaction form -- which is also why it generalises."""
    def __init__(self):
        super().__init__()
        self.input_dim = 139
        self.lin = nn.Linear(5, 1)

    def _feat(self, x):
        spr, tn, ds = x[:, SPRINT_DIM:SPRINT_DIM+1], x[:, 137:138], x[:, 138:139]
        return torch.cat([spr, tn, ds, spr*tn, spr*ds], dim=1)

    def forward(self, x):
        return self.lin(self._feat(x)).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


class TinyInteract(nn.Module):
    """Candidate #2. [spr,tn,ds] + all 3 pairwise products (6 features) ->
    16 hidden -> 1. Small non-linear extension; still cannot see content."""
    def __init__(self, hidden=16):
        super().__init__()
        self.input_dim = 139
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _feat(self, x):
        spr, tn, ds = x[:, SPRINT_DIM], x[:, 137], x[:, 138]
        return torch.stack([spr, tn, ds, spr*tn, spr*ds, tn*ds], dim=1)

    def forward(self, x):
        return self.net(self._feat(x)).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


class AdditiveSplit(nn.Module):
    """Candidate #3. PREFERRED SHIP if it passes.
    logit = f(content) + <phi(content), psi(context)>
    f() carries the marginal (content-only) signal. The dot-product term is
    the ONLY path by which context can affect the logit -- so any DiD > 0
    the gate measures is attributable to that term specifically, which is a
    clean story for the write-up. k=8 keeps the interaction subspace small
    relative to the ~200-step decisive cell."""
    def __init__(self, content_dim=CONTENT_DIM, k=8, hidden=64):
        super().__init__()
        self.input_dim = content_dim + 2
        self.f = nn.Sequential(nn.Linear(content_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))
        self.phi = nn.Linear(content_dim, k)
        self.psi = nn.Linear(2, k)

    def forward(self, x):
        content, context = x[:, :CONTENT_DIM], x[:, CONTENT_DIM:]
        f_out = self.f(content).squeeze(-1)
        inter = (self.phi(content) * self.psi(context)).sum(-1)
        return f_out + inter

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


def compute_topk_dims(expert_content, agent_content, k, always_include=(SPRINT_DIM,)):
    """|mean_E - mean_A| / pooled_sd, ranked descending. Returns k content-dim
    indices (0..136), guaranteed to include `always_include` (sprint is almost
    certainly top-ranked anyway given the 0.852 single-dim separability, but
    this makes it explicit rather than assumed)."""
    mE, mA = expert_content.mean(0), agent_content.mean(0)
    sd = np.sqrt(0.5 * (expert_content.var(0) + agent_content.var(0))) + 1e-9
    score = np.abs(mE - mA) / sd
    order = np.argsort(-score)
    keep = list(order[:k])
    for idx in always_include:
        if idx not in keep:
            keep = [idx] + keep[:-1]
    return sorted(set(keep)), score


class FeaturePrunedConcat(nn.Module):
    """Candidate #4. Top-k content dims (by separation score) + sprint
    (guaranteed) + context, standard 2-layer MLP. Fallback if AdditiveSplit
    underperforms -- keeps SOME spatial content but starves the
    137-dim overfitting route that killed concat/FiLM."""
    def __init__(self, keep_idx, hidden=64):
        super().__init__()
        self.register_buffer("keep_idx", torch.as_tensor(keep_idx, dtype=torch.long))
        d = len(keep_idx) + 2
        self.input_dim = 139  # accepts full input, slices internally
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        pruned = x[:, self.keep_idx]
        ctx = x[:, 137:139]
        h = torch.cat([pruned, ctx], dim=1)
        return self.net(h).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


class Bilinear(nn.Module):
    """Candidate #5. Content encoded to a small hidden space, then an
    explicit bilinear (low-rank) interaction with context on top of a
    content-only readout. r=8 rank cap keeps the interaction term from
    absorbing enough capacity to overfit content x context jointly."""
    def __init__(self, content_dim=CONTENT_DIM, hidden=64, r=8):
        super().__init__()
        self.input_dim = content_dim + 2
        self.enc = nn.Sequential(nn.Linear(content_dim, hidden), nn.ReLU())
        self.U = nn.Linear(hidden, r, bias=False)
        self.V = nn.Linear(2, r, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        content, context = x[:, :CONTENT_DIM], x[:, CONTENT_DIM:]
        h = self.enc(content)
        inter = (self.U(h) * self.V(context)).sum(-1)
        return self.out(h).squeeze(-1) + inter

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Candidate #6, negative controls. Same architectures as D1-D10 (already run
# and already failed) -- included here only so the screen table has them for
# direct comparison against the restricted family, on the SAME protocol
# (unstratified, no swap hinge) that #1-#5 use, which D1-D10 did not use
# uniformly.
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    def __init__(self, ctx=2, h=256, fh=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ctx, fh), nn.ReLU(), nn.Linear(fh, h*2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        g, b = self.net(c).chunk(2, dim=-1)
        return 1.0 + g, b


class ConcatDisc(nn.Module):
    def __init__(self, d=139, h=256):
        super().__init__()
        self.input_dim = d
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


class FiLMDisc(nn.Module):
    def __init__(self, content=CONTENT_DIM, ctx=2, h=256, fh=64):
        super().__init__()
        self.input_dim = content + ctx
        self.content = content
        self.l1 = nn.Linear(content, h); self.f1 = FiLMGenerator(ctx, h, fh)
        self.l2 = nn.Linear(h, h);       self.f2 = FiLMGenerator(ctx, h, fh)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        cont, c = x[:, :self.content], x[:, self.content:]
        g, b = self.f1(c); h = torch.relu(g * self.l1(cont) + b)
        g, b = self.f2(c); h = torch.relu(g * self.l2(h) + b)
        return self.out(h).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


def build_candidate(name, train_expert_content=None, train_agent_content=None):
    """Factory. train_*_content (N,137) numpy arrays required only for
    FeaturePrunedConcat's top-k selection -- pass None for other candidates."""
    if name == "logreg5":
        return LogReg5()
    if name == "tiny-interact":
        return TinyInteract()
    if name == "additive-split":
        return AdditiveSplit()
    if name == "feature-pruned-k8":
        keep, _ = compute_topk_dims(train_expert_content, train_agent_content, k=8)
        return FeaturePrunedConcat(keep)
    if name == "feature-pruned-k16":
        keep, _ = compute_topk_dims(train_expert_content, train_agent_content, k=16)
        return FeaturePrunedConcat(keep)
    if name == "feature-pruned-k32":
        keep, _ = compute_topk_dims(train_expert_content, train_agent_content, k=32)
        return FeaturePrunedConcat(keep)
    if name == "bilinear":
        return Bilinear()
    if name == "concat":
        return ConcatDisc()
    if name == "film":
        return FiLMDisc()
    raise ValueError(f"unknown candidate {name}")


CANDIDATE_ORDER = ["logreg5", "tiny-interact", "additive-split",
                  "feature-pruned-k8", "feature-pruned-k16", "feature-pruned-k32",
                  "bilinear", "concat", "film"]

class SprintOnlyDiscriminator(nn.Module):
    """The TRUE NC architecture. Input is [spr] alone -- structurally
    incapable of depending on T_norm or dScore, regardless of what
    training does. This replaces feeding TinyInteract a zeroed context,
    which only accidentally worked for that architecture's specific
    multiplicative feature form and isn't a principled guarantee for any
    other model class."""
    def __init__(self):
        super().__init__()
        self.input_dim = 139
        self.lin = nn.Linear(1, 1)

    def _feat(self, x):
        return x[:, SPRINT_DIM:SPRINT_DIM+1]

    def forward(self, x):
        return self.lin(self._feat(x)).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))

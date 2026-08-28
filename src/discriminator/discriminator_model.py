"""
discriminator_model.py

FiLM-conditioned discriminator. Rebuilt after the counterfactual swap-test
gate FAILED against the original concatenation architecture:
diagnose_context_sensitivity.py showed context got mid-pack first-layer
weight allocation but the second-lowest input-gradient of any block, and
failed the gate even on TRAIN data -- not a generalization gap, context
was simply diluted by 135 other dims that alone already gave ~96%
separability. This is the FiLM alternative named explicitly in the
original methodology (3.3.1) for exactly this failure mode.

Content (135 spatial/action dims) flows through the main MLP. Context
(2 dims: T_norm, delta_score) is fed to a small side-network per hidden
layer producing (gamma, beta), applied as:
    h' = ReLU(gamma * Linear(h) + beta)
-- modulation applied PRE-activation, the canonical FiLM placement. A
deliberate choice, not the only valid one.

FiLM generators are ZERO-INITIALIZED: at step 0, gamma=1 and beta=0
exactly, so a fresh FiLM discriminator is mathematically IDENTICAL to an
unconditioned network -- context has NO effect on the output until
training actually finds it useful. Verified explicitly in the test below,
not just asserted here.

Same external contract as before: forward() returns the PRE-SIGMOID
LOGIT, probability() applies sigmoid. Nothing outside this file changes.
"""

import torch
import torch.nn as nn

INPUT_DIM = 139
CONTEXT_DIM = 2
CONTENT_DIM = INPUT_DIM - CONTEXT_DIM  # 135
HIDDEN_DIM = 256
FILM_HIDDEN = 64


class FiLMGenerator(nn.Module):
    def __init__(self, context_dim=CONTEXT_DIM, hidden_dim=HIDDEN_DIM, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        out = self.net(context)
        raw_gamma, beta = out.chunk(2, dim=-1)
        return 1.0 + raw_gamma, beta  # gamma centered at 1 -> raw=0 gives gamma=1


"""class Discriminator(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 context_dim=CONTEXT_DIM, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.input_dim = input_dim
        self.content_dim = input_dim - context_dim

        self.linear1 = nn.Linear(self.content_dim, hidden_dim)
        self.film1 = FiLMGenerator(context_dim, hidden_dim, film_hidden)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMGenerator(context_dim, hidden_dim, film_hidden)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}), got {tuple(x.shape)}")

        content = x[:, :self.content_dim]
        context = x[:, self.content_dim:]

        h1 = self.linear1(content)
        gamma1, beta1 = self.film1(context)
        h1 = torch.relu(gamma1 * h1 + beta1)

        h2 = self.linear2(h1)
        gamma2, beta2 = self.film2(context)
        h2 = torch.relu(gamma2 * h2 + beta2)

        return self.linear3(h2).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))"""
class Discriminator(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 context_dim=CONTEXT_DIM, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.input_dim = input_dim
        self.content_dim = input_dim - context_dim

        self.linear1 = nn.Linear(self.content_dim, hidden_dim)
        self.film1 = FiLMGenerator(context_dim, hidden_dim, film_hidden)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMGenerator(context_dim, hidden_dim, film_hidden)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # NEW: auxiliary head, same h2 the real decision uses. Gradient
        # from this DOES flow back into film1/film2/linear1/linear2 --
        # that's the whole point, not a passive probe.
        self.context_pred_head = nn.Linear(hidden_dim, context_dim)

    def _shared_forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}), got {tuple(x.shape)}")
        content = x[:, :self.content_dim]
        context = x[:, self.content_dim:]

        h1 = self.linear1(content)
        gamma1, beta1 = self.film1(context)
        h1 = torch.relu(gamma1 * h1 + beta1)

        h2 = self.linear2(h1)
        gamma2, beta2 = self.film2(context)
        h2 = torch.relu(gamma2 * h2 + beta2)
        return h2

    def forward(self, x):
        return self.linear3(self._shared_forward(x)).squeeze(-1)

    def forward_with_context_pred(self, x):
        """(logits, context_pred), from the SAME h2. Used only by the
        auxiliary-loss path -- gate/eval code keeps calling forward()
        unchanged."""
        h2 = self._shared_forward(x)
        return self.linear3(h2).squeeze(-1), self.context_pred_head(h2)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))
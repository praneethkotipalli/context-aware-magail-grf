"""
discriminator_loss.py

Loss function for the MAGAIL-C discriminator D_phi.

Implements the locked components from Section 3.3.2 / Section 8:

  1. One-sided label smoothing (Salimans et al., 2016): expert=0.9, agent=0.1.

  2. Zero-centred R1 gradient penalty (Mescheder, Geiger & Nowozin, 2018),
     expert-only, on the pre-sigmoid logit.

Plus two NON-LOCKED, opt-in terms added while diagnosing the counterfactual
gate failure -- both default off except where noted, so DiscriminatorLoss()
with no extra args behaves like the original file.

  3. aux_loss (beta_aux, DEFAULT 0.0 -- OFF by default now): a side-head
     predicting (T_norm, delta_score) from h2. RESULT: proved context
     information reaches h2 (verified via diagnose_context_pred_vs_swap.py
     -- swapped-context predictions correctly tracked the injected target,
     not the true original), but had ~zero effect on the gate. Diagnosis:
     the aux head can learn its own private linear combination of h2,
     completely decoupled from whatever linear3 (the REAL/FAKE readout)
     uses. Proves information presence, does not force the classification
     readout to use it. Left in the code (harmless if re-enabled) but not
     the active fix.

  4. swap_consistency_loss (gamma_swap, DEFAULT 1.5 -- the active fix):
     operates on the literal forward() output -- true_logits =
     discriminator(real_x), swap_logits = discriminator(swap_context(real_x))
     -- the SAME path linear3 sits on, so there is no decoupling possible
     the way there was with the aux head. Hinge loss: only pushes while
     |P(true) - P(swapped)| < swap_margin, no gradient once satisfied.
     Directly trains the exact quantity the counterfactual gate measures,
     on TRAIN-split data only (held-out stays untouched -- the gate remains
     a genuine, uncontaminated generalization test).

     Populations to swap (swap_lw_features, swap_ll_features) are NOT
     computed in this file -- pass them in at construction, built ONCE
     from TRAIN-split features via select_by_true_context, same populations
     the gate itself tests, just restricted to train data. Each __call__
     draws a fresh random subsample of size swap_batch_size from each.

discriminator_model.py is assumed to expose:
    D_phi(x) -> logits, shape (B,) or (B,1)   [pre-sigmoid]
    D_phi.probability(x) -> sigmoid(logits)
    D_phi.forward_with_context_pred(x) -> (logits, context_pred)  [only
        called if beta_aux > 0]
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from context_swap import swap_context

# ---------------------------------------------------------------------------
DEFAULT_ETA = 1.0
DEFAULT_BETA_AUX = 0.0
# ---------------------------------------------------------------------------
# gamma_swap starting point. Reasoned, NOT tuned -- run
# check_swap_loss_magnitude.py before trusting this on a real run, same
# discipline as eta and beta_aux. At init, hinge ~= relu(0.15 - ~0.01) per
# direction (near-zero pre-existing sensitivity), summed over two
# directions -- 1.5 targets a meaningful-but-not-dominant share of total
# loss at that starting point. WILL need re-checking once sensitivity
# actually starts moving, since the hinge shrinks as the gate improves --
# unlike eta/beta_aux, this term is SUPPOSED to fade out as it succeeds.
DEFAULT_GAMMA_SWAP = 1.5
DEFAULT_SWAP_MARGIN = 0.15   # target |shift|, deliberately above the 0.1 gate
                               # threshold for buffer -- NOT itself the gate's
                               # own criterion, a training target only
DEFAULT_SWAP_BATCH_SIZE = 64


@dataclass
class DiscriminatorLossOutput:
    total_loss: torch.Tensor
    bce_expert: torch.Tensor
    bce_agent: torch.Tensor
    r1_penalty: torch.Tensor
    accuracy: torch.Tensor
    expert_accuracy: torch.Tensor
    agent_accuracy: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None
    swap_hinge_lw: Optional[torch.Tensor] = None   # late_winning -> late_losing direction
    swap_hinge_ll: Optional[torch.Tensor] = None   # late_losing -> late_winning direction
    swap_shift_lw: Optional[torch.Tensor] = None   # mean |shift| actually achieved, for logging
    swap_shift_ll: Optional[torch.Tensor] = None


class DiscriminatorLoss:
    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        expert_label: float = 0.9,
        agent_label: float = 0.1,
        beta_aux: float = DEFAULT_BETA_AUX,
        swap_lw_features: Optional[np.ndarray] = None,
        swap_ll_features: Optional[np.ndarray] = None,
        gamma_swap: float = DEFAULT_GAMMA_SWAP,
        swap_margin: float = DEFAULT_SWAP_MARGIN,
        swap_batch_size: int = DEFAULT_SWAP_BATCH_SIZE,
        swap_seed: int = 0,
    ):
        """
        swap_lw_features, swap_ll_features: (N, D) numpy arrays, TRAIN-split
            only (never held-out), already-normalized. Late-winning and
            late-losing populations respectively (same definitions
            context_shift_scoring.py's LATE_WINNING/LATE_LOSING use). Pass
            None (either or both) to disable that direction's term -- e.g.
            if a population happens to be empty for some data configuration,
            matching the on_empty='skip' philosophy elsewhere in this project.
            gamma_swap > 0 with both None means the term contributes 0,
            same as if gamma_swap were 0 -- fails soft, not loud, since this
            is a training-time convenience input, not a correctness-critical
            path the way the sampler's on_empty guard is.
        """
        if not (0.0 < agent_label < expert_label < 1.0):
            raise ValueError(
                f"Expected 0 < agent_label < expert_label < 1, "
                f"got agent_label={agent_label}, expert_label={expert_label}"
            )
        if beta_aux < 0 or gamma_swap < 0:
            raise ValueError(f"beta_aux and gamma_swap must be >= 0, got {beta_aux}, {gamma_swap}")

        self.eta = eta
        self.expert_label = expert_label
        self.agent_label = agent_label
        self.beta_aux = beta_aux

        self.swap_lw_features = swap_lw_features
        self.swap_ll_features = swap_ll_features
        self.gamma_swap = gamma_swap
        self.swap_margin = swap_margin
        self.swap_batch_size = swap_batch_size
        self._swap_rng = np.random.default_rng(swap_seed)

        self._bce = nn.BCEWithLogitsLoss()
        self._mse = nn.MSELoss()

    def r1_penalty(self, discriminator: nn.Module, expert_inputs: torch.Tensor):
        expert_inputs = expert_inputs.detach().clone().requires_grad_(True)
        logits = discriminator(expert_inputs)
        gradients = torch.autograd.grad(
            outputs=logits, inputs=expert_inputs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        grad_norm_sq = gradients.pow(2).sum(dim=1)
        r1 = (self.eta / 2.0) * grad_norm_sq.mean()
        return r1, logits

    def _context_prediction_loss(self, discriminator, expert_inputs, agent_inputs):
        _, expert_ctx_pred = discriminator.forward_with_context_pred(expert_inputs)
        _, agent_ctx_pred = discriminator.forward_with_context_pred(agent_inputs)
        expert_ctx_true = expert_inputs[:, -2:]
        agent_ctx_true = agent_inputs[:, -2:]
        return self._mse(expert_ctx_pred, expert_ctx_true) + self._mse(agent_ctx_pred, agent_ctx_true)

    def _swap_hinge(self, discriminator, features_np, target_t_norm, target_delta_score_raw):
        """Returns (hinge_loss, mean_abs_shift_achieved) or (None, None) if
        features_np is unavailable. Draws a FRESH random subsample every
        call -- over many training steps this cycles through the full
        train-split population, not just its first swap_batch_size rows."""
        if features_np is None or len(features_np) == 0:
            return None, None

        n = min(self.swap_batch_size, len(features_np))
        idx = self._swap_rng.choice(len(features_np), size=n, replace=False)
        batch = features_np[idx]
        swapped = np.stack([
            swap_context(row, target_t_norm, target_delta_score_raw) for row in batch
        ])

        true_t = torch.as_tensor(batch, dtype=torch.float32)
        swap_t = torch.as_tensor(swapped, dtype=torch.float32)

        true_logits = discriminator(true_t).view(-1)
        swap_logits = discriminator(swap_t).view(-1)

        true_probs = torch.sigmoid(true_logits)
        swap_probs = torch.sigmoid(swap_logits)

        shift = torch.abs(swap_probs - true_probs)
        hinge = torch.relu(self.swap_margin - shift).mean()
        return hinge, shift.mean().detach()

    def __call__(self, discriminator, expert_inputs, agent_inputs) -> DiscriminatorLossOutput:
        r1, expert_logits = self.r1_penalty(discriminator, expert_inputs)
        expert_logits = expert_logits.view(-1)
        agent_logits = discriminator(agent_inputs).view(-1)

        expert_targets = torch.full_like(expert_logits, self.expert_label)
        agent_targets = torch.full_like(agent_logits, self.agent_label)
        bce_expert = self._bce(expert_logits, expert_targets)
        bce_agent = self._bce(agent_logits, agent_targets)

        aux_loss_term = None
        aux_contribution = 0.0
        if self.beta_aux > 0:
            aux_loss_term = self._context_prediction_loss(discriminator, expert_inputs, agent_inputs)
            aux_contribution = self.beta_aux * aux_loss_term

        # late_winning -> late_losing: real target used by the gate/counterfactual_gate.py
        hinge_lw, shift_lw = self._swap_hinge(
            discriminator, self.swap_lw_features, target_t_norm=0.1, target_delta_score_raw=-2
        )
        # late_losing -> late_winning
        hinge_ll, shift_ll = self._swap_hinge(
            discriminator, self.swap_ll_features, target_t_norm=0.1, target_delta_score_raw=+2
        )

        swap_contribution = 0.0
        if self.gamma_swap > 0:
            if hinge_lw is not None:
                swap_contribution = swap_contribution + self.gamma_swap * hinge_lw
            if hinge_ll is not None:
                swap_contribution = swap_contribution + self.gamma_swap * hinge_ll

        total_loss = bce_expert + bce_agent + r1 + aux_contribution + swap_contribution

        with torch.no_grad():
            expert_probs = torch.sigmoid(expert_logits)
            agent_probs = torch.sigmoid(agent_logits)
            expert_acc = (expert_probs > 0.5).float().mean()
            agent_acc = (agent_probs < 0.5).float().mean()
            overall_acc = (expert_acc + agent_acc) / 2.0

        return DiscriminatorLossOutput(
            total_loss=total_loss,
            bce_expert=bce_expert.detach(),
            bce_agent=bce_agent.detach(),
            r1_penalty=r1.detach(),
            accuracy=overall_acc,
            expert_accuracy=expert_acc,
            agent_accuracy=agent_acc,
            aux_loss=aux_loss_term.detach() if aux_loss_term is not None else None,
            swap_hinge_lw=hinge_lw.detach() if hinge_lw is not None else None,
            swap_hinge_ll=hinge_ll.detach() if hinge_ll is not None else None,
            swap_shift_lw=shift_lw,
            swap_shift_ll=shift_ll,
        )
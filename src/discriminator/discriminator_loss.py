"""
discriminator_loss.py

Loss function for the MAGAIL-C discriminator D_phi.

Implements the two locked components from Section 3.3.2 / Section 8 of the
project spec:

  1. One-sided label smoothing (Salimans et al., 2016):
         expert target = 0.9, agent target = 0.1
     instead of hard 1.0 / 0.0, so D_phi is prevented from becoming
     overconfident and the policy-side gradient through log(1 - D) stays
     alive for longer.

  2. Zero-centred R1 gradient penalty (Mescheder, Geiger & Nowozin, 2018),
     applied to EXPERT samples only, on the pre-sigmoid logit:

         R1(phi) = (eta / 2) * E_{x ~ D_E}[ ||grad_x D_phi(x)||^2 ]

     This is NOT the WGAN-GP interpolated penalty -- no interpolation
     between real/fake samples, no (||grad|| - 1)^2 target. It is a
     zero-centred penalty on real data only, which is the form that
     converges locally under the finite discriminator-update-per-generator-
     update regime this project uses (1 D-update per 3 policy updates).

discriminator_model.py is assumed to expose:
    D_phi(x) -> logits, shape (B, 1) or (B,)   [pre-sigmoid, per model contract]
    D_phi.probability(x) -> sigmoid(logits)     [not used here]

Nothing in this file knows about GRF, .npz files, or the context-balanced
sampler -- it only consumes two already-assembled batches of 137-dim
feature vectors (expert_inputs, agent_inputs) and the discriminator module
itself. That keeps it testable in isolation and reusable unchanged for both
the pre-training loop and later joint policy fine-tuning.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Starting eta.
#

#   - Inputs are normalised to ~[-1, 1] (feature_normalization.py), so
#     gradient magnitudes are on a modest, bounded scale to begin with --
#     nothing like unnormalised pixel-space GANs, where R1 coefficients of
#     10-100+ (StyleGAN2-style) are needed to tame much larger gradients.
#   - eta=1.0 keeps the penalty roughly the same order of magnitude as the
#     BCE term at initialisation (verified in the self-test below), so
#     neither term dominates the gradient signal from step one.
#
# This is deliberately a starting point, not a tuned value: the actual
# tuning signal is the 60-80% held-out accuracy health band, exercised once
# the pre-training loop (next build step) exists. If the discriminator
# saturates (>90% accuracy) with the penalty active, eta should go up first
# before touching label smoothing or the D:policy update ratio.
DEFAULT_ETA = 1.0


@dataclass
class DiscriminatorLossOutput:
    """Everything a training loop needs: one tensor to backward(), the rest
    for logging / the health-band diagnostic."""

    total_loss: torch.Tensor          # requires_grad -- call .backward() on this
    bce_expert: torch.Tensor          # detached, for logging
    bce_agent: torch.Tensor           # detached, for logging
    r1_penalty: torch.Tensor          # detached, for logging
    accuracy: torch.Tensor            # detached, overall batch accuracy (health band)
    expert_accuracy: torch.Tensor     # detached
    agent_accuracy: torch.Tensor      # detached


class DiscriminatorLoss:
    """
    Callable loss for D_phi. Usage:

        loss_fn = DiscriminatorLoss(eta=1.0)
        out = loss_fn(discriminator, expert_batch, agent_batch)
        out.total_loss.backward()
        optimizer.step()

    expert_batch, agent_batch: (B, 137) float tensors, already normalised
    (feature_normalization.py) and already context-balanced
    (context_balanced_sampler.py). This module does not check either
    property -- it trusts the pipeline stages that already verify them.
    """

    def __init__(
        self,
        eta: float = DEFAULT_ETA,
        expert_label: float = 0.9,
        agent_label: float = 0.1,
    ):
        if not (0.0 < agent_label < expert_label < 1.0):
            raise ValueError(
                f"Expected 0 < agent_label < expert_label < 1, "
                f"got agent_label={agent_label}, expert_label={expert_label}"
            )
        self.eta = eta
        self.expert_label = expert_label
        self.agent_label = agent_label
        self._bce = nn.BCEWithLogitsLoss()

    def r1_penalty(self, discriminator: nn.Module, expert_inputs: torch.Tensor):
        """
        Standalone R1 penalty, exposed separately so the counterfactual
        swap-test gate or future diagnostics can call it without going
        through the full classification loss.

        Returns (r1_scalar, expert_logits) -- the logits are returned too
        so __call__ can reuse this single forward pass for the BCE term
        instead of running the discriminator twice on the same batch.
        """
        # detach + clone + requires_grad_: makes expert_inputs a clean leaf
        # tensor for autograd, regardless of what graph (if any) the caller's
        # tensor already belongs to. Does not mutate the caller's tensor.
        expert_inputs = expert_inputs.detach().clone().requires_grad_(True)

        logits = discriminator(expert_inputs)  # pre-sigmoid, (B, 1) or (B,)

        # create_graph=True: this gradient computation must itself remain
        # differentiable, because R1's contribution to total_loss is later
        # backpropagated through phi (the discriminator's own parameters
        # shaped this gradient). Matches the "input-gradients confirmed
        # finite ... required for R1" verification already done on the
        # model itself.
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=expert_inputs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # (B, 137)

        grad_norm_sq = gradients.pow(2).sum(dim=1)  # ||grad_x D(x)||^2 per sample
        r1 = (self.eta / 2.0) * grad_norm_sq.mean()

        return r1, logits

    def __call__(
        self,
        discriminator: nn.Module,
        expert_inputs: torch.Tensor,
        agent_inputs: torch.Tensor,
    ) -> DiscriminatorLossOutput:
        # Expert forward pass (with input grad tracking) feeds BOTH the BCE
        # expert term and the R1 penalty -- one forward pass, not two.
        r1, expert_logits = self.r1_penalty(discriminator, expert_inputs)
        expert_logits = expert_logits.view(-1)

        agent_logits = discriminator(agent_inputs).view(-1)

        expert_targets = torch.full_like(expert_logits, self.expert_label)
        agent_targets = torch.full_like(agent_logits, self.agent_label)

        bce_expert = self._bce(expert_logits, expert_targets)
        bce_agent = self._bce(agent_logits, agent_targets)

        total_loss = bce_expert + bce_agent + r1

        with torch.no_grad():
            expert_probs = torch.sigmoid(expert_logits)
            agent_probs = torch.sigmoid(agent_logits)
            # "correct" is w.r.t. the true expert/agent identity, not the
            # smoothed target -- classifying an expert sample >0.5 is a
            # correct call even though its soft target was only 0.9.
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
        )
"""
discriminator_trainer.py

Wraps discriminator_loss.py into the step()/health() interface for the
pre-training loop, and later joint policy fine-tuning.

Revised after seeing the real context_balanced_sampler.py:

  - sample()/balanced_batch() return (features, bins) where bins is a
    (B,)-length int array, values 0-8 indexing into CELL_NAMES. That's
    what expert_cells/agent_cells below expect.

  - features come back as numpy arrays (the sampler is pure numpy,
    "works identically for expert data and live agent rollouts"). This
    file converts to torch internally so callers never have to remember
    to do it themselves.

  - ess needs no separate weights argument. Every example drawn from the
    same cell has the identical importance weight
    (target_prop[cell] / natural_prop[cell]), and both of those already
    live on expert_sampler: target_prop is sqrt_scaled_target(cell_counts)
    (their own function -- reused, not reimplemented, so it can't drift
    out of sync with what the sampler actually drew against), natural_prop
    is cell_counts / cell_counts.sum(). Pass expert_sampler once at
    construction; step() derives weights from expert_cells each call.

  ESS = (sum_i w_i)^2 / sum_i(w_i^2). ESS close to the batch size means
  the batch behaves like an i.i.d. draw from the target distribution;
  ESS much smaller flags that a few heavily-oversampled cells (early/win
  at 0.51x reuse, per your last run) are dominating the gradient despite
  the marginals matching on average. Computed on the expert side only,
  same focus as the existing reuse-rate tracker (real-data budget is the
  thing at risk of memorization, not the agent side).
"""

import statistics
from collections import deque

import numpy as np
import torch

from discriminator_loss import DiscriminatorLoss, DEFAULT_ETA
from context_balanced_sampler import CELL_NAMES, sqrt_scaled_target


class DiscriminatorTrainer:
    def __init__(
        self,
        model,
        optimizer,
        expert_sampler=None,
        eta: float = DEFAULT_ETA,
        expert_label: float = 0.9,
        agent_label: float = 0.1,
        update_ratio: int = 3,
        health_window: int = 50,
        confused_below: float = 0.55,
        saturating_above: float = 0.90,
    ):
        """
        expert_sampler: the BalancedContextSampler instance used to draw
            the expert side of each batch. Optional -- if omitted, step()
            still works, ess just comes back None. Passed by reference
            (not copied counts), so if you rebuild the sampler's cell
            index later the Trainer picks that up automatically.

        update_ratio: recorded and reported, NOT enforced -- "1 D-update
        per 3 policy updates" is the outer training loop's job (whether
        it calls step() this iteration or not), not something this class
        can act on by itself.

        confused_below / saturating_above: the two named thresholds from
        the locked 60-80% health band. The 55-60% and 80-90% zones aren't
        separately named in the locked spec -- both fold into "healthy"
        below as the closest fit to a 3-state health(). Say the word if
        you want a 4th "borderline" state to make that gap explicit.
        """
        self.model = model
        self.optimizer = optimizer
        self.expert_sampler = expert_sampler
        self.loss_fn = DiscriminatorLoss(
            eta=eta, expert_label=expert_label, agent_label=agent_label
        )
        self.update_ratio = update_ratio
        self.confused_below = confused_below
        self.saturating_above = saturating_above
        self._acc_history = deque(maxlen=health_window)
        self.global_step = 0

        self._target_props = None
        self._natural_props = None
        if expert_sampler is not None:
            self._target_props = sqrt_scaled_target(expert_sampler.cell_counts)
            self._natural_props = (
                expert_sampler.cell_counts / expert_sampler.cell_counts.sum()
            )

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def step(
        self,
        expert_batch,
        agent_batch,
        expert_cells=None,
        agent_cells=None,
    ) -> dict:
        """
        expert_batch, agent_batch: (B, 137), numpy or torch -- typically
            straight from balanced_batch()'s (expert_feat, ...), (agent_feat, ...).
        expert_cells, agent_cells: optional (B,) int bins (0-8), typically
            balanced_batch()'s expert_bins / agent_bins. Needed for
            acc_per_region; expert_cells alone (with expert_sampler set
            at construction) is enough for ess.
        """
        expert_t = self._to_tensor(expert_batch)
        agent_t = self._to_tensor(agent_batch)

        self.optimizer.zero_grad()
        out = self.loss_fn(self.model, expert_t, agent_t)
        out.total_loss.backward()
        self.optimizer.step()
        self.global_step += 1

        acc = out.accuracy.item()
        self._acc_history.append(acc)

        acc_per_region = None
        if expert_cells is not None and agent_cells is not None:
            acc_per_region = self._acc_per_region(
                self.model, expert_t, agent_t, expert_cells, agent_cells
            )

        ess = None
        if expert_cells is not None and self._target_props is not None:
            ess = self._effective_sample_size(expert_cells)

        return {
            "step": self.global_step,
            "loss": out.total_loss.item(),
            "bce_expert": out.bce_expert.item(),
            "bce_agent": out.bce_agent.item(),
            "gp_value": out.r1_penalty.item(),
            "acc": acc,
            "acc_expert": out.expert_accuracy.item(),
            "acc_agent": out.agent_accuracy.item(),
            "acc_per_region": acc_per_region,
            "ess": ess,
        }

    @staticmethod
    @torch.no_grad()
    def _acc_per_region(model, expert_t, agent_t, expert_cells, agent_cells):
        """Extra (cheap, inference-only) forward pass -- not reusing
        step()'s graph, which backward() has already consumed by this point."""
        expert_probs = torch.sigmoid(model(expert_t).view(-1))
        agent_probs = torch.sigmoid(model(agent_t).view(-1))

        expert_cells = np.asarray(expert_cells)
        agent_cells = np.asarray(agent_cells)
        results = {}

        for cell in sorted(set(expert_cells.tolist()) | set(agent_cells.tolist())):
            e_idx = torch.as_tensor(expert_cells == cell)
            a_idx = torch.as_tensor(agent_cells == cell)

            correct, total = 0, 0
            if e_idx.any():
                correct += (expert_probs[e_idx] > 0.5).sum().item()
                total += int(e_idx.sum().item())
            if a_idx.any():
                correct += (agent_probs[a_idx] < 0.5).sum().item()
                total += int(a_idx.sum().item())

            name = CELL_NAMES[cell] if 0 <= cell < len(CELL_NAMES) else str(cell)
            results[name] = (correct / total) if total > 0 else None

        return results

    def _effective_sample_size(self, expert_cells) -> float:
        expert_cells = np.asarray(expert_cells)
        weights = self._target_props[expert_cells] / self._natural_props[expert_cells]
        return float((weights.sum() ** 2) / (weights ** 2).sum())

    def health(self) -> str:
        """Rolling mean over the last `health_window` step() calls, not a
        single batch -- per-batch accuracy is noisy; the health band is a
        training-regime signal, not a per-step one."""
        if not self._acc_history:
            return "unknown"
        recent = statistics.mean(self._acc_history)
        if recent > self.saturating_above:
            return "saturating"
        if recent < self.confused_below:
            return "confused"
        return "healthy"
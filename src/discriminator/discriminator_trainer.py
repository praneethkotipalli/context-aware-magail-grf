"""
discriminator_trainer.py

Wraps discriminator_loss.py into the step()/health() interface.

Revised to thread through the new swap-consistency term (swap_lw_features,
swap_ll_features, gamma_swap, swap_margin) and surface its metrics in
step()'s return dict: swap_hinge_lw/ll (the loss terms themselves) and
swap_shift_lw/ll (the actual mean|shift| achieved that batch -- THIS is
the number to watch converge toward >0.1, the gate's real criterion).

Everything else unchanged from the prior version.
"""

import statistics
from collections import deque

import numpy as np
import torch

from discriminator_loss import DiscriminatorLoss, DEFAULT_ETA, DEFAULT_BETA_AUX, DEFAULT_GAMMA_SWAP, DEFAULT_SWAP_MARGIN
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
        beta_aux: float = DEFAULT_BETA_AUX,
        swap_lw_features=None,
        swap_ll_features=None,
        gamma_swap: float = DEFAULT_GAMMA_SWAP,
        swap_margin: float = DEFAULT_SWAP_MARGIN,
        update_ratio: int = 3,
        health_window: int = 50,
        confused_below: float = 0.55,
        saturating_above: float = 0.90,
    ):
        """
        swap_lw_features, swap_ll_features: TRAIN-split late-winning /
            late-losing populations (numpy arrays), built ONCE before
            constructing this Trainer -- see phase_a_pretraining.py /
            phase_b_pretraining.py for how they're built via
            select_by_true_context on the train split. Pass None to
            disable the swap-consistency term entirely (gamma_swap has
            no effect either way in that case).
        """
        self.model = model
        self.optimizer = optimizer
        self.expert_sampler = expert_sampler
        self.loss_fn = DiscriminatorLoss(
            eta=eta, expert_label=expert_label, agent_label=agent_label,
            beta_aux=beta_aux,
            swap_lw_features=swap_lw_features, swap_ll_features=swap_ll_features,
            gamma_swap=gamma_swap, swap_margin=swap_margin,
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
    def get_recent_accuracy(self):
        """Rolling mean accuracy over the health window. Used by the
        finetune loop to throttle discriminator updates."""
        if not self._acc_history:
            return None
        return statistics.mean(self._acc_history)
    
    def step(self, expert_batch, agent_batch, expert_cells=None, agent_cells=None) -> dict:
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
            "aux_loss": out.aux_loss.item() if out.aux_loss is not None else None,
            "swap_hinge_lw": out.swap_hinge_lw.item() if out.swap_hinge_lw is not None else None,
            "swap_hinge_ll": out.swap_hinge_ll.item() if out.swap_hinge_ll is not None else None,
            "swap_shift_lw": out.swap_shift_lw.item() if out.swap_shift_lw is not None else None,
            "swap_shift_ll": out.swap_shift_ll.item() if out.swap_shift_ll is not None else None,
            "acc": acc,
            "acc_expert": out.expert_accuracy.item(),
            "acc_agent": out.agent_accuracy.item(),
            "acc_per_region": acc_per_region,
            "ess": ess,
        }

    @staticmethod
    @torch.no_grad()
    def _acc_per_region(model, expert_t, agent_t, expert_cells, agent_cells):
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
        if not self._acc_history:
            return "unknown"
        recent = statistics.mean(self._acc_history)
        if recent > self.saturating_above:
            return "saturating"
        if recent < self.confused_below:
            return "confused"
        return "healthy"
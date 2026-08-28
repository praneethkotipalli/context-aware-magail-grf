"""
oracle_discriminator.py

Hard-coded (NOT trained -- no gradient descent, no optimizer) discriminator
used ONLY to establish a defensible CEILING for the counterfactual
swap-test gate, given the actual data. Run this, then compare its result
directly against the trained FiLM checkpoint's result on the SAME gate.

D*(x,c) is the closed-form Bayes-optimal classifier under EXACTLY ONE
feature: whether action_captured == SPRINT (index 13 of the 19-way
one-hot). Everything else in the 137-dim vector is ignored entirely.
This is deliberately the best ANY discriminator could do using only
sprint frequency as a context signal -- your own MECHA analysis found no
reliable formation-based signal, so SAP is the strongest known real
correlate, making this a genuine ceiling estimate, not an arbitrary one.

p_expert_sprint[cell] / p_agent_sprint[cell]: computed ONCE from real
data, per 9-cell taxonomy region (classify_bin, imported from
build_expert_dataset.py -- not reimplemented). Expert side uses the
TRAIN split ONLY (same seed=0 episode-level split already used for the
real discriminator and gate) -- held-out data stays genuinely unseen,
same discipline as everywhere else in this project. Agent side uses the
full mappo_features_cache.npz (no held-out concept on that side, same
as the real training runs).

D*(x,c) = p_e / (p_e + p_a)   =>   logit = log(p_e) - log(p_a)
(algebraically identical to log(D*/(1-D*)), just simpler to compute).

INTERPRETATION:
  - If the oracle FAILS the same gate the trained discriminator failed,
    the 0.1 mean-shift threshold is unrealistic for this population,
    independent of architecture -- the threshold is the problem, not FiLM.
  - If the oracle PASSES, architecture/training is still the real
    bottleneck, and projection (or more training, or an auxiliary
    context-prediction head) is worth pursuing.
"""

import numpy as np
import torch
import torch.nn as nn

from build_expert_dataset import classify_bin
from context_balanced_sampler import CELL_NAMES, N_CELLS
from feature_derivation import BLOCK_SLICES
from held_out_split import make_episode_split, split_features_by_episode
from counterfactual_gate import run_counterfactual_gate, print_gate_report

SPRINT_FEATURE_IDX = BLOCK_SLICES['action'].start + 13  # 116 + 13 = 129
CONTEXT_SLICE = BLOCK_SLICES['context']


def build_cell_sprint_rates(features, bins):
    """Laplace-smoothed P(sprint | cell), one per of the 9 cells.
    Fully vectorized -- uses the cache's own precomputed bins column,
    no per-row classify_bin calls needed here."""
    is_sprint = features[:, SPRINT_FEATURE_IDX] > 0.5
    rates = np.zeros(N_CELLS)
    counts = np.zeros(N_CELLS, dtype=int)
    for c in range(N_CELLS):
        mask = bins == c
        n_total = mask.sum()
        n_sprint = (mask & is_sprint).sum()
        counts[c] = n_total
        rates[c] = (n_sprint + 1.0) / (n_total + 2.0)  # Laplace-1 smoothing
    return rates, counts


class OracleDiscriminator(nn.Module):
    def __init__(self, p_expert_sprint, p_agent_sprint):
        super().__init__()
        # buffers, not parameters -- nothing here is ever optimized
        self.register_buffer("p_expert_sprint", torch.as_tensor(p_expert_sprint, dtype=torch.float32))
        self.register_buffer("p_agent_sprint", torch.as_tensor(p_agent_sprint, dtype=torch.float32))

    def forward(self, x):
        """x: (B, 137). Returns (B,) pre-sigmoid logits, same contract as
        the real Discriminator. Loops over rows to call classify_bin --
        batches here are at most a few thousand (gate populations), so
        this is a diagnostic-scale cost, not a training-loop cost."""
        x_np = x.detach().cpu().numpy()
        ctx = x_np[:, CONTEXT_SLICE]
        is_sprint = x_np[:, SPRINT_FEATURE_IDX] > 0.5

        logits = np.zeros(x_np.shape[0], dtype=np.float32)
        for i in range(x_np.shape[0]):
            cell = classify_bin(float(ctx[i, 0]), float(ctx[i, 1]))
            p_e = self.p_expert_sprint[cell].item()
            p_a = self.p_agent_sprint[cell].item()
            if not is_sprint[i]:
                p_e, p_a = 1.0 - p_e, 1.0 - p_a
            logits[i] = np.log(p_e) - np.log(p_a)

        return torch.as_tensor(logits, dtype=torch.float32)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


def run():
    print("Building oracle from real data...\n")

    expert_cache = np.load("expert_features_cache.npz", allow_pickle=True)
    train_ep, held_ep = make_episode_split(expert_cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        expert_cache["features"], expert_cache["bins"], expert_cache["episode_ids"], train_ep, held_ep
    )
    p_expert_sprint, expert_counts = build_cell_sprint_rates(train_feat, train_bins)

    agent_cache = np.load("mappo_features_cache.npz")
    p_agent_sprint, agent_counts = build_cell_sprint_rates(agent_cache["features"], agent_cache["bins"])

    print(f"{'cell':12s} {'expert n':>9s} {'P(sprint|expert)':>17s} {'agent n':>9s} {'P(sprint|agent)':>16s}")
    for c in range(N_CELLS):
        print(f"  {CELL_NAMES[c]:10s} {expert_counts[c]:9d} {p_expert_sprint[c]*100:16.2f}% "
              f"{agent_counts[c]:9d} {p_agent_sprint[c]*100:15.2f}%")

    # sanity check on the "agent SAP is roughly context-independent" assumption
    agent_spread = p_agent_sprint.max() - p_agent_sprint.min()
    print(f"\n  agent SAP spread across cells: {agent_spread*100:.2f}pp "
          f"({'roughly constant -- assumption holds' if agent_spread < 0.05 else 'NOT constant -- worth a closer look'})")

    oracle = OracleDiscriminator(p_expert_sprint, p_agent_sprint)

    print("\nRunning the SAME gate the trained discriminator was evaluated on, "
          "on the SAME held-out split, against the ORACLE:\n")
    overall_pass, summary = run_counterfactual_gate(oracle, held_feat)
    print_gate_report(overall_pass, summary)

    print("\n" + "=" * 70)
    if overall_pass:
        print("ORACLE PASSES both directions.")
        print("-> The 0.1 threshold IS achievable given this data.")
        print("-> Architecture/training is the real bottleneck, not the threshold.")
        print("-> Worth pursuing: more training steps, an auxiliary context-")
        print("   prediction loss, or the projection-discriminator alternative.")
    else:
        print("ORACLE FAILS at least one direction, using ONLY the sprint signal")
        print("(the strongest known real correlate -- MECHA showed no reliable")
        print("formation signal).")
        print("-> The threshold, not the architecture, is the likely constraint")
        print("   for whichever direction failed.")
        print("-> Worth reconsidering: a per-direction threshold, or treating")
        print("   the 0.1 figure as needing recalibration against this specific")
        print("   population, rather than continuing to rebuild the discriminator.")
    print("=" * 70)


if __name__ == '__main__':
    run()
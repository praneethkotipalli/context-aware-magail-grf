"""smoke_test_gate_on_real_data.py"""
import numpy as np
import torch.nn as nn

from context_shift_scoring import select_by_true_context, LATE_WINNING, EARLY_LOSING
from counterfactual_gate import run_counterfactual_gate, print_gate_report

cache = np.load("expert_features_cache.npz")
features = cache["features"]

mask_lw = select_by_true_context(
    features, t_norm_max=LATE_WINNING['t_norm_max'], delta_score_sign=LATE_WINNING['delta_score_sign']
)
mask_el = select_by_true_context(
    features, t_norm_min=EARLY_LOSING['t_norm_min'], delta_score_sign=EARLY_LOSING['delta_score_sign']
)
print(f"real late-winning examples (t_norm<=0.3, delta_score>0): {mask_lw.sum()}")
print(f"real early-losing examples  (t_norm>=0.7, delta_score<0): {mask_el.sum()}")

class StubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(137, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.net(x)

disc = StubDiscriminator()
overall, summary = run_counterfactual_gate(disc, features)
print_gate_report(overall, summary)
print("\n(untrained discriminator -- this verdict is meaningless, this is a plumbing check only)")
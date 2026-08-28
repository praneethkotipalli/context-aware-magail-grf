"""test_counterfactual_gate.py"""
"""import numpy as np
import torch.nn as nn

from counterfactual_gate import run_counterfactual_gate, print_gate_report
from feature_derivation import BLOCK_SLICES

INPUT_DIM = 137
CTX_SLICE = BLOCK_SLICES['context']


class MixedSensitivityModel(nn.Module):
    ""logit = marker * delta_score_norm * scale, else 0. `marker` lives
    in feature column 0 (repurposed here only, for full test control) --
    lets a test specify exactly which rows the model is/isn't sensitive
    to, independent of the actual context values used for selection.""
    def __init__(self, scale=10.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        marker = x[:, 0:1]
        delta = x[:, CTX_SLICE][:, 1:2]
        return marker * delta * self.scale


def make_population(n, t_norm_range, delta_score_range, frac_sensitive, seed):
    rng = np.random.default_rng(seed)
    feats = rng.uniform(-1, 1, size=(n, INPUT_DIM)).astype(np.float32)
    t_norm = rng.uniform(*t_norm_range, size=n)
    delta_score = rng.uniform(*delta_score_range, size=n)
    feats[:, CTX_SLICE] = np.stack([t_norm, delta_score], axis=1)
    n_sensitive = int(n * frac_sensitive)
    marker = np.zeros(n)
    marker[:n_sensitive] = 1.0
    feats[:, 0] = marker
    return feats


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


disc = MixedSensitivityModel()

print("Scenario 1: fully context-sensitive both directions -> clean PASS, no warnings")
lw = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=1.0, seed=1)
el = make_population(40, (0.7, 1.0), (-1.0, -0.1), frac_sensitive=1.0, seed=2)
features1 = np.concatenate([lw, el])
overall1, summary1 = run_counterfactual_gate(disc, features1)
print_gate_report(overall1, summary1)
check("scenario 1: overall PASS", overall1)
check("scenario 1: no consistency warnings", len(summary1["warnings"]) == 0)

print("\nScenario 2: fully context-blind both directions -> clean FAIL")
lw2 = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=0.0, seed=3)
el2 = make_population(40, (0.7, 1.0), (-1.0, -0.1), frac_sensitive=0.0, seed=4)
features2 = np.concatenate([lw2, el2])
overall2, summary2 = run_counterfactual_gate(disc, features2)
check("scenario 2: overall FAIL", overall2 is False)
check("scenario 2: both directions individually fail",
      not summary2["directions"][0]["passes_mean_criterion"]
      and not summary2["directions"][1]["passes_mean_criterion"])

print("\nScenario 3: sensitive one direction, blind the other -> FAIL overall")
lw3 = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=1.0, seed=5)   # sensitive
el3 = make_population(40, (0.7, 1.0), (-1.0, -0.1), frac_sensitive=0.0, seed=6)  # blind
features3 = np.concatenate([lw3, el3])
overall3, summary3 = run_counterfactual_gate(disc, features3)
check("scenario 3: overall FAIL despite one direction passing", overall3 is False)
check("scenario 3: late_winning->early_losing passes",
      summary3["directions"][0]["passes_mean_criterion"])
check("scenario 3: early_losing->late_winning fails",
      not summary3["directions"][1]["passes_mean_criterion"])

print("\nScenario 4: 30% sensitive in one direction -> passes mean, triggers consistency warning")
lw4 = make_population(60, (0.0, 0.3), (0.1, 1.0), frac_sensitive=0.3, seed=7)
el4 = make_population(40, (0.7, 1.0), (-1.0, -0.1), frac_sensitive=1.0, seed=8)
features4 = np.concatenate([lw4, el4])
overall4, summary4 = run_counterfactual_gate(disc, features4)
print_gate_report(overall4, summary4)
check("scenario 4: overall PASS (mean criterion met in both directions)", overall4)
check("scenario 4: exactly one warning raised", len(summary4["warnings"]) == 1)
check("scenario 4: warning is about the late_winning direction",
      "late_winning -> early_losing" in summary4["warnings"][0])

print("\nScenario 5: no matching examples -> raises ValueError")
mid_only = make_population(40, (0.4, 0.6), (-0.2, 0.2), frac_sensitive=1.0, seed=9)
try:
    run_counterfactual_gate(disc, mid_only)
    check("scenario 5: should have raised ValueError", False)
except ValueError as e:
    check(f"scenario 5: correctly raised ValueError ({e})", True)

print("\nAll checks passed.")"""
import numpy as np
import torch.nn as nn

from counterfactual_gate import run_counterfactual_gate, print_gate_report
from feature_derivation import BLOCK_SLICES

INPUT_DIM = 139
CTX_SLICE = BLOCK_SLICES['context']


class MixedSensitivityModel(nn.Module):
    """logit = marker * delta_score_norm * scale, else 0. `marker` lives
    in feature column 0 (repurposed here only, for full test control) --
    lets a test specify exactly which rows the model is/isn't sensitive
    to, independent of the actual context values used for selection."""
    def __init__(self, scale=10.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        marker = x[:, 0:1]
        delta = x[:, CTX_SLICE][:, 1:2]
        return marker * delta * self.scale


def make_population(n, t_norm_range, delta_score_range, frac_sensitive, seed):
    rng = np.random.default_rng(seed)
    feats = rng.uniform(-1, 1, size=(n, INPUT_DIM)).astype(np.float32)
    t_norm = rng.uniform(*t_norm_range, size=n)
    delta_score = rng.uniform(*delta_score_range, size=n)
    feats[:, CTX_SLICE] = np.stack([t_norm, delta_score], axis=1)
    n_sensitive = int(n * frac_sensitive)
    marker = np.zeros(n)
    marker[:n_sensitive] = 1.0
    feats[:, 0] = marker
    return feats


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


disc = MixedSensitivityModel()

print("Scenario 1: fully context-sensitive both directions -> clean PASS, no warnings")
lw = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=1.0, seed=1)
ll = make_population(40, (0.0, 0.3), (-1.0, -0.1), frac_sensitive=1.0, seed=2)
features1 = np.concatenate([lw, ll])
overall1, summary1 = run_counterfactual_gate(disc, features1)
print_gate_report(overall1, summary1)
check("scenario 1: overall PASS", overall1)
check("scenario 1: no consistency warnings", len(summary1["warnings"]) == 0)

print("\nScenario 2: fully context-blind both directions -> clean FAIL")
lw2 = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=0.0, seed=3)
ll2 = make_population(40, (0.0, 0.3), (-1.0, -0.1), frac_sensitive=0.0, seed=4)
features2 = np.concatenate([lw2, ll2])
overall2, summary2 = run_counterfactual_gate(disc, features2)
check("scenario 2: overall FAIL", overall2 is False)
check("scenario 2: both directions individually fail",
      not summary2["directions"][0]["passes_mean_criterion"]
      and not summary2["directions"][1]["passes_mean_criterion"])

print("\nScenario 3: sensitive one direction, blind the other -> FAIL overall")
lw3 = make_population(40, (0.0, 0.3), (0.1, 1.0), frac_sensitive=1.0, seed=5)   # sensitive
ll3 = make_population(40, (0.0, 0.3), (-1.0, -0.1), frac_sensitive=0.0, seed=6)  # blind
features3 = np.concatenate([lw3, ll3])
overall3, summary3 = run_counterfactual_gate(disc, features3)
check("scenario 3: overall FAIL despite one direction passing", overall3 is False)
check("scenario 3: late_winning->late_losing passes",
      summary3["directions"][0]["passes_mean_criterion"])
check("scenario 3: late_losing->late_winning fails",
      not summary3["directions"][1]["passes_mean_criterion"])

print("\nScenario 4: 30% sensitive in one direction -> passes mean, triggers consistency warning")
lw4 = make_population(60, (0.0, 0.3), (0.1, 1.0), frac_sensitive=0.3, seed=7)
ll4 = make_population(40, (0.0, 0.3), (-1.0, -0.1), frac_sensitive=1.0, seed=8)
features4 = np.concatenate([lw4, ll4])
overall4, summary4 = run_counterfactual_gate(disc, features4)
print_gate_report(overall4, summary4)
check("scenario 4: overall PASS (mean criterion met in both directions)", overall4)
check("scenario 4: exactly one warning raised", len(summary4["warnings"]) == 1)
check("scenario 4: warning is about the late_winning direction",
      "late_winning -> late_losing" in summary4["warnings"][0])

print("\nScenario 5: no matching examples -> raises ValueError")
mid_only = make_population(40, (0.4, 0.6), (-0.2, 0.2), frac_sensitive=1.0, seed=9)
try:
    run_counterfactual_gate(disc, mid_only)
    check("scenario 5: should have raised ValueError", False)
except ValueError as e:
    check(f"scenario 5: correctly raised ValueError ({e})", True)

print("\nAll checks passed.")
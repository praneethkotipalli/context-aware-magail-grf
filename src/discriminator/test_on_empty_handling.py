"""test_on_empty_handling.py"""
import numpy as np
from context_balanced_sampler import BalancedContextSampler, balanced_batch, N_CELLS, CELL_NAMES

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

rng = np.random.default_rng(0)

# Build a sampler with cells 0 and 8 deliberately empty (mimics
# early/win + late/draw being empty, same shape as the real random-policy data)
n = 1000
features = rng.standard_normal((n, 137)).astype(np.float32)
bins = rng.integers(1, 8, size=n)  # only cells 1-7 populated, 0 and 8 empty
sampler = BalancedContextSampler(features, bins, name="test_sampler")

check("cell 0 confirmed empty in this synthetic sampler", sampler.cell_counts[0] == 0)
check("cell 8 confirmed empty in this synthetic sampler", sampler.cell_counts[8] == 0)

print("\n1. Default behavior (on_empty='raise') unchanged")
counts_hitting_empty = np.zeros(N_CELLS, dtype=int)
counts_hitting_empty[0] = 5  # request from the empty cell
raised = False
try:
    sampler.sample(counts_hitting_empty, rng)
except ValueError:
    raised = True
check("default on_empty='raise' still raises on empty cell", raised)

print("\n2. on_empty='skip' draws nothing from empty cells, no error")
feat, cell_bins = sampler.sample(counts_hitting_empty, rng, on_empty='skip')
check("skip mode: no exception raised", True)
check("skip mode: batch has 0 rows (only empty cell was requested)", feat.shape[0] == 0)

counts_mixed = np.zeros(N_CELLS, dtype=int)
counts_mixed[0] = 5   # empty -- should contribute 0
counts_mixed[3] = 10  # populated -- should contribute 10
feat2, cell_bins2 = sampler.sample(counts_mixed, rng, on_empty='skip')
check(f"skip mode: batch has exactly 10 rows, not 15 (got {feat2.shape[0]})", feat2.shape[0] == 10)
check("skip mode: all returned rows are from cell 3, none from cell 0",
      np.all(cell_bins2 == 3))

print("\n3. balanced_batch realigns both sides when a cell is empty on only ONE side")
# expert has cell 0 populated, agent doesn't -- realistic mismatch
expert_bins = rng.integers(0, 9, size=500)   # all 9 cells present
agent_bins = rng.integers(1, 8, size=500)     # cells 0, 8 empty on agent side
expert_feat = rng.standard_normal((500, 137)).astype(np.float32)
agent_feat = rng.standard_normal((500, 137)).astype(np.float32)

expert_sampler = BalancedContextSampler(expert_feat, expert_bins, name="expert")
agent_sampler = BalancedContextSampler(agent_feat, agent_bins, name="agent")

target = np.full(N_CELLS, 1.0 / N_CELLS)  # uniform target for simplicity
(e_feat, e_bins), (a_feat, a_bins) = balanced_batch(
    expert_sampler, agent_sampler, batch_size=90, target_props=target, rng=rng, on_empty='skip'
)

check("realigned: expert side has NO examples from cell 0 (agent couldn't supply it)",
      0 not in e_bins)
check("realigned: expert side has NO examples from cell 8 (agent couldn't supply it)",
      8 not in e_bins)
check("realigned: expert and agent cell sets are IDENTICAL after realignment",
      set(e_bins.tolist()) == set(a_bins.tolist()))
print(f"  cells present after realignment: {sorted(set(e_bins.tolist()))}")
print(f"  expert batch size: {len(e_bins)}, agent batch size: {len(a_bins)}")
check("both sides same size after realignment", len(e_bins) == len(a_bins))

print("\nAll checks passed.")
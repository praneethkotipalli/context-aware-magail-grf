## Entry 1 — Environment architecture: two-conda-environment split

What was built/decided: Established a two-environment architecture: dissertation-grf (Python 3.9, PyTorch 1.13, GRF/Light-MALib compatible) for environment stepping and baseline checkpoint inference, and dissertation-train (Python 3.10, modern PyTorch) reserved for new components (discriminator, fine-tuning loop) once built.

Why this approach (vs. alternatives considered): A single shared environment was considered first but rejected: GRF/Light-MALib's dependency chain (gym==0.21.0, older ray, torch==1.13) is fundamentally incompatible with the modern PyTorch/CUDA versions needed for efficient training on newer hardware. Attempting to force a single environment to satisfy both would either break GRF or force training-side components onto an outdated, unsupported PyTorch version.

Verification performed: Confirmed both environments could be created independently and that GRF-dependent code ran correctly under dissertation-grf without needing any package from the training-side environment.

Related debugging log entry: N/A (architectural decision made proactively, before any conflict was hit)

What this demonstrates: A deliberate, upfront architectural choice was made based on anticipated dependency conflicts rather than discovered reactively through failure — a design decision, not a fix.

Entry 2  — Baseline metric definitions locked

What was built/decided: Formal, locked definitions for the four core evaluation metrics: Sprint Action Percentage (SAP), Sprint Bout Frequency (SBF), Mean Episode Convex Hull Area (MECHA, possession-conditional), and Centroid Variance (CV, computed as sum of per-axis variances rather than a naive flattened variance).

Why this approach (vs. alternatives considered):

MECHA restricted to possession timesteps only, rather than all timesteps: defensive compression is tactically correct play and should not be penalised as poor formation; an all-timestep metric would conflate offensive spread with defensive shape.
CV computed as var(x) + var(y) (trace of the covariance matrix) rather than np.var() on the flattened 2D array: the latter would implicitly weight the x-axis more heavily, since the GRF pitch is roughly 2.4x longer than it is wide, biasing the metric toward horizontal movement.
SAP/SBF built on the sticky-action time series rather than the discrete action array (see Debugging Log Entry 1) — this was a correction partway through, not an original design choice.

Verification performed: Unit tests against synthetic data (perfectly spread vs clustered formations for MECHA; alternating vs constant sprint states for SAP/SBF; stationary vs moving positions for CV) confirming each metric responds in the theoretically expected direction. Independent closed-form statistical validation for SAP/SBF (see Debugging Log Entry 3).

Related debugging log entry: Entry 1 (sprint index correction), Entry 3 (statistical validation).

What this demonstrates: Metric definitions were not adopted casually from convention but were each individually justified against a specific alternative, with the reasoning tied to the actual football/tactical semantics being measured — directly supporting the Methodology chapter's evaluation framework section.

Entry 3  — Baseline agent selection methodology

What was built/decided: A structured, three-criterion baseline selection process across nine candidate pre-trained 5v5 policies: (1) architectural compatibility filtering (via desc.pkl inspection, reducing to six viable candidates), (2) exploratory small-sample comparison (n=10, then n=30) to narrow to four competent candidates, (3) full statistical characterisation (n=500 per policy) to make the final selection.

Why this approach (vs. alternatives considered): An arbitrary single-checkpoint selection (e.g., picking based on folder name or the framework's own radar-plot visualisation alone) was rejected in favour of direct, quantitative measurement using the dissertation's own locked metrics, since the framework's published comparison uses a different, undocumented statistic set (see Methodology chapter, baseline selection justification) not directly reducible to SAP/MECHA/win-rate. A staged small-sample-then-full-sample approach was used rather than running all nine candidates at n=500 immediately, to conserve compute time given the two architecture families discovered mid-process.

Verification performed: Final n=500 selection (PassingMain_v2: win_rate=75.4%, SAP=93.28%±1.76%) was checked for consistency against the pattern observed at every earlier smaller sample size (n=4, n=10, n=30), all of which agreed on PassingMain_v2 as the strongest villain candidate — providing informal but meaningful robustness evidence despite the final run not using an explicitly fixed random seed.

Related debugging log entry: Entry 4 (architecture mismatch discovered during this process), Entry 5 (multiprocessing failure during the n=30 comparison run).

What this demonstrates: Baseline selection was treated as a genuine empirical question requiring its own evidence, rather than an arbitrary convenience choice — directly supporting the "select and justify choice of approach" language in Criterion 4, and providing a concrete example of escalating rigour (staged sample sizes) as a deliberate resource-management strategy under time constraint.

Entry 4  — Parallel execution architecture for baseline characterisation

What was built/decided: Task-chunked multiprocessing architecture for the full-scale (n=500 x 4 policies) baseline run: episodes split into fixed-size chunks distributed across a bounded worker pool (rather than one process per policy), each chunk writing its own crash-safe incremental CSV, live progress reported via a shared queue, W&B logging restricted to the main process only.

Why this approach (vs. alternatives considered): A naive "one process per candidate policy" design (used in the initial exploratory comparison) wastes available cores when the number of policies is small relative to available cores (4 policies vs 24 cores on the production machine); task-chunking decouples parallelism from the number of experimental conditions, keeping all cores utilised regardless of how many policies/conditions are being compared. Single-process-per-policy was also the direct cause of the OOM failure documented in Debugging Log Entry 5, reinforcing the need for a bounded-pool design.

Verification performed: Small-scale smoke test (4 episodes/policy, 2-episode chunks, 4 workers) run and confirmed correct before committing to the full-scale (500 episodes/policy, 25-episode chunks, 20 workers) production run; resource usage (CPU spread, memory headroom) checked via htop during the smoke test.

Related debugging log entry: Entry 5 (the OOM/oversubscription failure that motivated this redesign).

What this demonstrates: Infrastructure was iteratively improved based on a real failure, tested at small scale before being trusted at production scale, and the final design (task-chunking) is directly reusable for the forthcoming ablation study (5 conditions × 5 seeds), where the same "many independent short-to-medium tasks distributed across a worker pool" pattern applies directly — this reusability is itself evidence of deliberate, forward-looking engineering rather than one-off scripting.

Entry 5  — Cross-machine reproducibility (laptop → Blackwell)

What was built/decided: Full environment and pipeline replication from the development machine (laptop) to a shared university lab machine (24-core, 48GB VRAM), including conda environment recreation, GRF_MARL re-cloning with Git LFS, and re-running the full verification chain (encoder dimension check, live checkpoint loading, end-to-end inference test) before trusting any results generated on the new machine.

Why this approach (vs. alternatives considered): Copying results or trusting "it worked on the laptop, it'll work here" without re-verification was explicitly rejected; several issues (a pandas/numpy ABI incompatibility, hardcoded laptop-specific file paths in verification scripts, gym packaging metadata rejected by a newer pip version) were only caught because the full verification chain was re-run rather than assumed to transfer.

Verification performed: Full re-verification chain: environment creation, GRF import and environment-stepping test, sticky-action regression test against the theoretical SAP prediction, checkpoint loading as a live object, encoder output dimension check, and end-to-end single-inference test — all re-confirmed passing on the new machine before any baseline data was generated there.

Related debugging log entry: Entry 6 (pandas/numpy ABI mismatch); additional undocumented path and packaging issues encountered during this process, several resolved via targeted pip/setuptools version pinning.

What this demonstrates: Reproducibility across compute environments was treated as something to actively verify rather than assume, directly relevant to the dissertation's eventual multi-machine ablation study (Discussion: parallel execution across multiple identical lab machines) and demonstrating awareness that "identical hardware" does not guarantee "identical software state" without explicit verification.

## Entry 6  — Context injection verification suite

**What was built/decided:** A staged verification notebook confirming, in order: 
(1) context extraction formula (T_norm, delta_score) against raw GRF observations, 
(2) shape correctness for the 194-dim context-extended input at both single-sample 
and batched scale, (3) bit-for-bit equivalence between the frozen reference policy 
called directly on 192-dim input versus called via the build-then-strip pathway 
used during fine-tuning, (4) discriminator skeleton shape correctness, and 
(5) a counterfactual context-swap test confirming context dimensions structurally 
influence discriminator output even prior to any training.

**Why this approach:** Each structural assumption underlying the context-aware 
MAGAIL architecture was tested in isolation, on synthetic or minimal real data, 
before being combined into the full training pipeline -- following the same 
verify-before-trust discipline established during baseline development (see 
Debugging Log entries 1-6), applied proactively here rather than reactively 
after a failure.

**Verification performed:** All five checks passed on first or second attempt 
(one import-path correction needed for the pi* loading test, consistent with the 
`light_malib` import pattern already identified in Debugging Log Entry — 
checkpoint loading requires the full GRF_MARL_ROOT on sys.path, not just the 
encoder submodule path).

**What this demonstrates:** The context-injection mechanism -- the core novel 
component of this dissertation -- was verified structurally sound before any 
real training data or learned discriminator weights existed, reducing the risk 
that a later training failure would be confounded by an undetected architectural 
bug in the context pathway itself.


# Implementation Log — Discriminator Build, This Session

Covers everything built from `discriminator_loss.py` through the FiLM-rebuilt,
gate-failed checkpoint. Starting state: discriminator model + context-balanced
sampler already existed and were verified (prior session). Ending state:
discriminator architecturally complete, both pre-training phases run twice
(original + FiLM), counterfactual gate still FAILING — checkpoint exists for
throughput/pilot purposes only, NOT validated for real training.

---

## 1. `discriminator_loss.py`
BCEWithLogitsLoss + one-sided label smoothing (expert=0.9, agent=0.1) +
zero-centred R1 gradient penalty (expert-only, pre-sigmoid logit, `(η/2)·E[‖∇_x D‖²]`).
Single forward pass on the expert batch feeds both the BCE-expert term and R1.
`DEFAULT_ETA = 1.0`, explicitly provisional, not imported from AMP.
**Verified:** `test_discriminator_loss.py`, 6/6 checks, including R1 matching a
hand-computed value exactly (0.6850) via `ConstantGradModel`.

## 2. `discriminator_trainer.py`
Wraps loss into `step()`/`health()`. `step()` returns loss, bce_expert/agent,
gp_value, acc (+ per-side), acc_per_region, ess. `health()`: >90% saturating,
<55% confused, else healthy (rolling window).
- First draft assumed a separate `expert_weights` arg for ESS — wrong.
- Corrected after seeing real `context_balanced_sampler.py`: numpy→torch
  bridging added (sampler is pure numpy); ESS derived from
  `expert_sampler.cell_counts` + the sampler's own `sqrt_scaled_target()`
  (reused, not reimplemented); `acc_per_region` keyed by real `CELL_NAMES`.
**Verified:** `test_discriminator_trainer.py`, 5 groups — numpy bridging,
ESS exact-match on a provable single-cell case, ESS vs. independent
recomputation on mixed cells, CELL_NAMES keying, health() thresholds.

## 3. Agent-side pipeline closure (§5.4 from prior handoff)
- Fixed hardcoded Blackwell username in `record_baseline_rollout.py`
  (later regressed — see Debugging Log #1).
- Confirmed `actor.pt` (374501 bytes) is a real checkpoint, not an LFS stub.
- Ran on the laptop (not Blackwell) — pure inference, no GRF-render dependency.
- Built `check_active_player_mapping.py` — confirmed `raw_obs[i]['active']`
  returns `[1,2,3,4]` for `i=[0,1,2,3]`, i.e. `raw_obs[i] ↔ left_team[i+1]`
  exactly, across all 4 agents (original script only checked agent 0).
- Re-ran `batch_verify_features.py` (as `batch_verify_features_baseline.py`)
  against agent data — clean, 0 NaN/Inf, 0 out-of-range, 15,000 steps.

## 4. Counterfactual swap-test gate (Section 3.3.3)
Three files:
- **`context_swap.py`** — swaps only the context slot (dims 135–136),
  applies the same `clip(-3,3)/3` scaling as `feature_normalization.py`.
- **`context_shift_scoring.py`** — `select_by_true_context()` +
  `score_context_shift()`, locked `LATE_WINNING`/`EARLY_LOSING` constants
  matching the (corrected) T_norm-decreasing convention.
- **`counterfactual_gate.py`** — bidirectional (late_win→early_loss AND
  early_loss→late_win), each direction gated independently on mean|shift|>0.1
  (locked threshold), fraction-exceeding-0.1 reported+flagged but not gating
  (explicit user decision).
**Verified:** each piece individually, then `counterfactual_gate.py` across
5 synthetic scenarios (clean pass, clean fail, one-direction-blind, the
mean-passes/median-zero consistency-warning case, empty-population
ValueError). Smoke-tested against the real 156,052-step cache with an
untrained stub — correctly meaningless/near-zero, confirmed plumbing only.

## 5. Corpus finalization checks
- Confirmed `build_expert_dataset.py`'s `classify_bin()` already used the
  correct (post-bug-fix) T_norm-decreasing direction — no rebuild needed.
- Discovered `rebuild_all_from_raw.py` was never actually needed for the
  9-cell `bins` array specifically (it recomputes fresh from raw fields each
  run) — only relevant for the separate 4-category `context_region` text
  field in audit JSONs, which the discriminator pipeline never reads.
- **Found:** written dissertation Section 3.1/3.2.2 still states the
  *original* (pre-bug-fix) T_norm-increasing direction — opposite of every
  piece of code in this project. Flagged for the corrections pass, not yet
  fixed in the chapter text.
- Re-confirmed `feature_derivation.py` uses the identical `steps_left/3001.0`
  denominator as `build_expert_dataset.py` — no drift between bin label and
  embedded feature context value.
- Re-ran `build_expert_dataset.py` against the final 52-episode corpus —
  identical 156,052-step cache, identical cell distribution to before
  (confirms nothing needed rebuilding, corpus was already correctly cached).

## 6. Episode-level held-out split (decision + build)
Decision made explicitly (not step-level): steps within an episode are
correlated, not independent — same principle as the project's own
episode-cluster CSI bootstrap. Step-level splitting would let the model
"generalize" to a near-duplicate of a training example.
- `build_expert_dataset.py` extended: now also saves `episode_ids` (per-step),
  `episode_outcomes` (per-episode, derived from final score), `episode_filenames`.
  Re-run confirmed outcome counts match exactly: 14W/13D/25L.
- **`held_out_split.py`** — `make_episode_split()` (stratified by outcome,
  per-group rounding), `split_features_by_episode()`.
**Verified:** `test_held_out_split.py` — 41/11 episode split, 3W/3D/5L held-out
(proportional to 14/13/25), all 9 context cells still present in the 11
held-out episodes despite being whole matches, no train/held-out overlap.

## 7. Random-policy rollout (Phase A data source)
- `check_action_space.py` — confirmed `env.action_space` is
  `MultiDiscrete([19,19,19,19])`, `sample()` returns a directly-usable
  4-array, no per-agent looping needed (simpler than the actor-based script).
- **`record_random_policy_rollout.py`** — structurally identical to
  `record_baseline_rollout.py` with the actor/encoder block replaced by
  `env.action_space.sample()`. No model dependency at all.
- **`build_agent_dataset_random_policy.py`** — mirrors
  `build_agent_dataset_mappo.py` pattern, imports `classify_bin` rather than
  redefining it. Run against 5 episodes: 4 cells empty
  (early/win, mid/win, late/win, late/draw) — matches independently-computed
  `agent_cell_histogram.py` exactly.

## 8. Empty-cell handling (decision + implementation)
Decision made explicitly: full 9-cell target for both phases, "let it be" —
don't force cells that genuinely have no data, don't drop them permanently either.
- `context_balanced_sampler.py`: added `on_empty` param to `sample()`
  (default `'raise'`, preserves all prior tested behavior) and threaded
  through `balanced_batch()`. `on_empty='skip'` draws 0 from an empty cell
  rather than erroring; `balanced_batch` additionally realigns both sides to
  their common cell set when a cell is empty on only one side, preserving
  the marginal-matching guarantee.
**Verified:** `test_on_empty_handling.py` — default behavior unchanged,
skip-mode draws exactly the right count, and critically: after realignment
expert/agent cell sets are IDENTICAL (this is the check that actually
protects the leakage guarantee).

## 9. Phase A pre-training (original concat architecture)
`phase_a_pretraining.py` — 10,000 steps vs. random-policy, batch=128
(matches the pre-documented ~24x worst-cell-reuse figure), W&B every 100
steps, held-out eval every 500 (monitoring only, not per-phase-gating —
combined-gate interpretation of the locked spec, explicit decision).
**Result:** health `healthy`→`saturating` at step 800, stayed saturating
92% of run. Final held-out acc 0.9760. Flagged as *expected* — Phase A has
no adversarial policy loop, saturation here doesn't carry the same risk it
would during real joint fine-tuning.

## 10. Frozen-MAPPO rollout expansion (Phase B data source)
- Decision: 100 additional episodes (105 total), explicit user choice
  ("100+, closest to real reuse parity").
- **`build_agent_dataset_mappo.py`** — same pattern as random-policy builder.
  Cell distribution at 105 episodes: `late/draw` fully resolved (25,641
  steps — NOT structurally rare after all, just needed volume); `early/loss`
  still thin (2,058 steps) but no longer empty.

## 11. Phase B pre-training (original concat architecture)
`phase_b_pretraining.py` — resumes from Phase A checkpoint (not fresh init),
SAME held-out split seed as Phase A, agent side now `mappo_features_cache.npz`.
**Result:** health stayed `healthy` for all 10,000 steps — genuinely different
regime from Phase A's saturation, interpreted as evidence the harder
human-vs-competent-but-inhuman discrimination task didn't collapse to a
trivial shortcut. Combined held-out acc 0.9625, gate criterion (0.75) PASSED
on this metric. `acc_expert`/`acc_agent` gap widened vs. Phase A
(0.930/0.734) — flagged as plausible given MAPPO plays genuine football,
not noise.

## 12. Real counterfactual gate — FIRST RUN — FAILED
Against the Phase B (concat-architecture) checkpoint, on held-out expert data.
Both directions: mean|shift| ≈ 0.003–0.004, 0% of examples exceeding 0.1.

## 13. Diagnosis
**`diagnose_context_sensitivity.py`** — three checks: per-block input-gradient
magnitude, per-block first-layer weight norm, gate re-run on TRAIN data.
**Findings:** context got mid-pack weight allocation (0.64, between opp_vel
0.39 and action 0.995) — not parameter-starved. Input-gradient for context
lowest of any block but only ~2x below average, not order-of-magnitude —
not consistent with active R1 suppression. Gate FAILED even on train data —
not a generalization gap, context was never learned at all. Conclusion:
dilution — 135 other dims already gave ~96% separability, no optimization
pressure ever reached for the 2 context dims.

## 14. FiLM rebuild (`discriminator_model.py`)
Per Section 3.3.1's own pre-registered alternative for exactly this failure
mode. Context (2 dims) generates per-layer (γ, β) via a small side-network,
applied as `h' = ReLU(γ·Linear(h) + β)` pre-activation, to both hidden layers.
FiLM generators zero-initialized (`γ=1, β=0` at init) — a fresh model is
mathematically identical to unconditioned, context has zero effect until
training finds it useful.
External contract unchanged (`forward()`→logit, `.probability()`→sigmoid) —
zero changes required to loss/trainer/sampler/gate files, only the model swaps.
**Verified:** `test_discriminator_film.py` — 167,809 params (hand-computed,
matches exactly), all standard checks pass, PLUS a new FiLM-specific check:
same content + different context → max output difference 0.00000000 at init,
confirming the zero-init claim empirically rather than asserting it.

## 15. Phase A + Phase B rerun (FiLM architecture, fresh init)
Both re-run from scratch (old checkpoints incompatible — different state
dict shape). `phase_a_pretraining.py`/`phase_b_pretraining.py` required
zero code changes — both only ever imported `Discriminator` and called
`forward()`/`probability()`.
**Phase A result:** held-out acc 0.9983 (final). **Phase B result:** held-out
acc 0.9690, combined gate PASS on the accuracy metric.

## 16. Real counterfactual gate — SECOND RUN — STILL FAILED, but improved
| | concat (original) | FiLM | change |
|---|---|---|---|
| late_win→early_loss mean\|shift\| | 0.0037 | 0.0073 | ~2x |
| early_loss→late_win mean\|shift\| | 0.0043 | 0.0523 | ~12x |
| frac exceeding 0.1 | 0% / 0% | 0% / 2.8% | first examples ever cross |

FiLM gave the network genuine capacity to use context (confirmed, not
assumed — the numbers moved substantially). Still failing: no training
*incentive* to use that capacity, since (a) real correlational signal
already exists elsewhere (action-block sprint frequency correlates with
true context in human data, per the project's own SAP statistics), and
(b) a larger, context-*independent* shortcut dominates — the baseline's
near-constant ~93.5% SAP vs. demonstrators' lower SAP in essentially every
context region, which alone approaches the observed ~96% separability
without the network needing to reason about context at all.

## 17. Decision point — stopped here for tonight
Confirmed: no literal feature duplication (steps_left/scores feed only the
context slot, nowhere else in the 137 dims). Checkpoint from step 15/16
explicitly marked: **exists, does not pass the gate, usable for pilot
throughput/pipeline-wiring purposes only — not validated for real ablation
training.** Remedy options identified but not yet built: (a) extend training
steps and see if the FiLM trend continues, (b) add an auxiliary
context-prediction loss head to force representational pressure toward
context-awareness. Neither pursued yet — deferred to next session, since
throughput measurement doesn't require a gate-passing discriminator.

# Implementation Log — Discriminator Build, This Session

Covers everything built from `discriminator_loss.py` through the FiLM-rebuilt,
gate-failed checkpoint. Starting state: discriminator model + context-balanced
sampler already existed and were verified (prior session). Ending state:
discriminator architecturally complete, both pre-training phases run twice
(original + FiLM), counterfactual gate still FAILING — checkpoint exists for
throughput/pilot purposes only, NOT validated for real training.

---

## 1. `discriminator_loss.py`
BCEWithLogitsLoss + one-sided label smoothing (expert=0.9, agent=0.1) +
zero-centred R1 gradient penalty (expert-only, pre-sigmoid logit, `(η/2)·E[‖∇_x D‖²]`).
Single forward pass on the expert batch feeds both the BCE-expert term and R1.
`DEFAULT_ETA = 1.0`, explicitly provisional, not imported from AMP.
**Verified:** `test_discriminator_loss.py`, 6/6 checks, including R1 matching a
hand-computed value exactly (0.6850) via `ConstantGradModel`.

## 2. `discriminator_trainer.py`
Wraps loss into `step()`/`health()`. `step()` returns loss, bce_expert/agent,
gp_value, acc (+ per-side), acc_per_region, ess. `health()`: >90% saturating,
<55% confused, else healthy (rolling window).
- First draft assumed a separate `expert_weights` arg for ESS — wrong.
- Corrected after seeing real `context_balanced_sampler.py`: numpy→torch
  bridging added (sampler is pure numpy); ESS derived from
  `expert_sampler.cell_counts` + the sampler's own `sqrt_scaled_target()`
  (reused, not reimplemented); `acc_per_region` keyed by real `CELL_NAMES`.
**Verified:** `test_discriminator_trainer.py`, 5 groups — numpy bridging,
ESS exact-match on a provable single-cell case, ESS vs. independent
recomputation on mixed cells, CELL_NAMES keying, health() thresholds.

## 3. Agent-side pipeline closure (§5.4 from prior handoff)
- Fixed hardcoded Blackwell username in `record_baseline_rollout.py`
  (later regressed — see Debugging Log #1).
- Confirmed `actor.pt` (374501 bytes) is a real checkpoint, not an LFS stub.
- Ran on the laptop (not Blackwell) — pure inference, no GRF-render dependency.
- Built `check_active_player_mapping.py` — confirmed `raw_obs[i]['active']`
  returns `[1,2,3,4]` for `i=[0,1,2,3]`, i.e. `raw_obs[i] ↔ left_team[i+1]`
  exactly, across all 4 agents (original script only checked agent 0).
- Re-ran `batch_verify_features.py` (as `batch_verify_features_baseline.py`)
  against agent data — clean, 0 NaN/Inf, 0 out-of-range, 15,000 steps.

## 4. Counterfactual swap-test gate (Section 3.3.3)
Three files:
- **`context_swap.py`** — swaps only the context slot (dims 135–136),
  applies the same `clip(-3,3)/3` scaling as `feature_normalization.py`.
- **`context_shift_scoring.py`** — `select_by_true_context()` +
  `score_context_shift()`, locked `LATE_WINNING`/`EARLY_LOSING` constants
  matching the (corrected) T_norm-decreasing convention.
- **`counterfactual_gate.py`** — bidirectional (late_win→early_loss AND
  early_loss→late_win), each direction gated independently on mean|shift|>0.1
  (locked threshold), fraction-exceeding-0.1 reported+flagged but not gating
  (explicit user decision).
**Verified:** each piece individually, then `counterfactual_gate.py` across
5 synthetic scenarios (clean pass, clean fail, one-direction-blind, the
mean-passes/median-zero consistency-warning case, empty-population
ValueError). Smoke-tested against the real 156,052-step cache with an
untrained stub — correctly meaningless/near-zero, confirmed plumbing only.

## 5. Corpus finalization checks
- Confirmed `build_expert_dataset.py`'s `classify_bin()` already used the
  correct (post-bug-fix) T_norm-decreasing direction — no rebuild needed.
- Discovered `rebuild_all_from_raw.py` was never actually needed for the
  9-cell `bins` array specifically (it recomputes fresh from raw fields each
  run) — only relevant for the separate 4-category `context_region` text
  field in audit JSONs, which the discriminator pipeline never reads.
- **Found:** written dissertation Section 3.1/3.2.2 still states the
  *original* (pre-bug-fix) T_norm-increasing direction — opposite of every
  piece of code in this project. Flagged for the corrections pass, not yet
  fixed in the chapter text.
- Re-confirmed `feature_derivation.py` uses the identical `steps_left/3001.0`
  denominator as `build_expert_dataset.py` — no drift between bin label and
  embedded feature context value.
- Re-ran `build_expert_dataset.py` against the final 52-episode corpus —
  identical 156,052-step cache, identical cell distribution to before
  (confirms nothing needed rebuilding, corpus was already correctly cached).

## 6. Episode-level held-out split (decision + build)
Decision made explicitly (not step-level): steps within an episode are
correlated, not independent — same principle as the project's own
episode-cluster CSI bootstrap. Step-level splitting would let the model
"generalize" to a near-duplicate of a training example.
- `build_expert_dataset.py` extended: now also saves `episode_ids` (per-step),
  `episode_outcomes` (per-episode, derived from final score), `episode_filenames`.
  Re-run confirmed outcome counts match exactly: 14W/13D/25L.
- **`held_out_split.py`** — `make_episode_split()` (stratified by outcome,
  per-group rounding), `split_features_by_episode()`.
**Verified:** `test_held_out_split.py` — 41/11 episode split, 3W/3D/5L held-out
(proportional to 14/13/25), all 9 context cells still present in the 11
held-out episodes despite being whole matches, no train/held-out overlap.

## 7. Random-policy rollout (Phase A data source)
- `check_action_space.py` — confirmed `env.action_space` is
  `MultiDiscrete([19,19,19,19])`, `sample()` returns a directly-usable
  4-array, no per-agent looping needed (simpler than the actor-based script).
- **`record_random_policy_rollout.py`** — structurally identical to
  `record_baseline_rollout.py` with the actor/encoder block replaced by
  `env.action_space.sample()`. No model dependency at all.
- **`build_agent_dataset_random_policy.py`** — mirrors
  `build_agent_dataset_mappo.py` pattern, imports `classify_bin` rather than
  redefining it. Run against 5 episodes: 4 cells empty
  (early/win, mid/win, late/win, late/draw) — matches independently-computed
  `agent_cell_histogram.py` exactly.

## 8. Empty-cell handling (decision + implementation)
Decision made explicitly: full 9-cell target for both phases, "let it be" —
don't force cells that genuinely have no data, don't drop them permanently either.
- `context_balanced_sampler.py`: added `on_empty` param to `sample()`
  (default `'raise'`, preserves all prior tested behavior) and threaded
  through `balanced_batch()`. `on_empty='skip'` draws 0 from an empty cell
  rather than erroring; `balanced_batch` additionally realigns both sides to
  their common cell set when a cell is empty on only one side, preserving
  the marginal-matching guarantee.
**Verified:** `test_on_empty_handling.py` — default behavior unchanged,
skip-mode draws exactly the right count, and critically: after realignment
expert/agent cell sets are IDENTICAL (this is the check that actually
protects the leakage guarantee).

## 9. Phase A pre-training (original concat architecture)
`phase_a_pretraining.py` — 10,000 steps vs. random-policy, batch=128
(matches the pre-documented ~24x worst-cell-reuse figure), W&B every 100
steps, held-out eval every 500 (monitoring only, not per-phase-gating —
combined-gate interpretation of the locked spec, explicit decision).
**Result:** health `healthy`→`saturating` at step 800, stayed saturating
92% of run. Final held-out acc 0.9760. Flagged as *expected* — Phase A has
no adversarial policy loop, saturation here doesn't carry the same risk it
would during real joint fine-tuning.

## 10. Frozen-MAPPO rollout expansion (Phase B data source)
- Decision: 100 additional episodes (105 total), explicit user choice
  ("100+, closest to real reuse parity").
- **`build_agent_dataset_mappo.py`** — same pattern as random-policy builder.
  Cell distribution at 105 episodes: `late/draw` fully resolved (25,641
  steps — NOT structurally rare after all, just needed volume); `early/loss`
  still thin (2,058 steps) but no longer empty.

## 11. Phase B pre-training (original concat architecture)
`phase_b_pretraining.py` — resumes from Phase A checkpoint (not fresh init),
SAME held-out split seed as Phase A, agent side now `mappo_features_cache.npz`.
**Result:** health stayed `healthy` for all 10,000 steps — genuinely different
regime from Phase A's saturation, interpreted as evidence the harder
human-vs-competent-but-inhuman discrimination task didn't collapse to a
trivial shortcut. Combined held-out acc 0.9625, gate criterion (0.75) PASSED
on this metric. `acc_expert`/`acc_agent` gap widened vs. Phase A
(0.930/0.734) — flagged as plausible given MAPPO plays genuine football,
not noise.

## 12. Real counterfactual gate — FIRST RUN — FAILED
Against the Phase B (concat-architecture) checkpoint, on held-out expert data.
Both directions: mean|shift| ≈ 0.003–0.004, 0% of examples exceeding 0.1.

## 13. Diagnosis
**`diagnose_context_sensitivity.py`** — three checks: per-block input-gradient
magnitude, per-block first-layer weight norm, gate re-run on TRAIN data.
**Findings:** context got mid-pack weight allocation (0.64, between opp_vel
0.39 and action 0.995) — not parameter-starved. Input-gradient for context
lowest of any block but only ~2x below average, not order-of-magnitude —
not consistent with active R1 suppression. Gate FAILED even on train data —
not a generalization gap, context was never learned at all. Conclusion:
dilution — 135 other dims already gave ~96% separability, no optimization
pressure ever reached for the 2 context dims.

## 14. FiLM rebuild (`discriminator_model.py`)
Per Section 3.3.1's own pre-registered alternative for exactly this failure
mode. Context (2 dims) generates per-layer (γ, β) via a small side-network,
applied as `h' = ReLU(γ·Linear(h) + β)` pre-activation, to both hidden layers.
FiLM generators zero-initialized (`γ=1, β=0` at init) — a fresh model is
mathematically identical to unconditioned, context has zero effect until
training finds it useful.
External contract unchanged (`forward()`→logit, `.probability()`→sigmoid) —
zero changes required to loss/trainer/sampler/gate files, only the model swaps.
**Verified:** `test_discriminator_film.py` — 167,809 params (hand-computed,
matches exactly), all standard checks pass, PLUS a new FiLM-specific check:
same content + different context → max output difference 0.00000000 at init,
confirming the zero-init claim empirically rather than asserting it.

## 15. Phase A + Phase B rerun (FiLM architecture, fresh init)
Both re-run from scratch (old checkpoints incompatible — different state
dict shape). `phase_a_pretraining.py`/`phase_b_pretraining.py` required
zero code changes — both only ever imported `Discriminator` and called
`forward()`/`probability()`.
**Phase A result:** held-out acc 0.9983 (final). **Phase B result:** held-out
acc 0.9690, combined gate PASS on the accuracy metric.

## 16. Real counterfactual gate — SECOND RUN — STILL FAILED, but improved
| | concat (original) | FiLM | change |
|---|---|---|---|
| late_win→early_loss mean\|shift\| | 0.0037 | 0.0073 | ~2x |
| early_loss→late_win mean\|shift\| | 0.0043 | 0.0523 | ~12x |
| frac exceeding 0.1 | 0% / 0% | 0% / 2.8% | first examples ever cross |

FiLM gave the network genuine capacity to use context (confirmed, not
assumed — the numbers moved substantially). Still failing: no training
*incentive* to use that capacity, since (a) real correlational signal
already exists elsewhere (action-block sprint frequency correlates with
true context in human data, per the project's own SAP statistics), and
(b) a larger, context-*independent* shortcut dominates — the baseline's
near-constant ~93.5% SAP vs. demonstrators' lower SAP in essentially every
context region, which alone approaches the observed ~96% separability
without the network needing to reason about context at all.

## 17a. Real actor observation space discovered mid-session (policy-side, not discriminator, but logged here since found during discriminator/policy handoff)
`enhanced_LightActionMask_5.FeatureEncoder` — actual `observation_space` is
**192-dim** (`133+59`, confirmed via `Linear(in_features=192,...)` in the
loaded actor + `PartialLayernorm`'s two LayerNorm groups matching exactly),
NOT the 115-dim `simple115_v2` Section 3.1 states. More consequentially:
`match_state` (part of the 59-dim group) already contains
`steps_left/3001` (= T_norm, identical convention) and a clipped score
ratio (= ΔScore, differently scaled) — **the frozen baseline was never
architecturally blind to time or score**, contra Section 3.1's founding
claim ("this absence is not incidental... induces state aliasing").
**Reframing needed for the corrections pass:** the real story is a
reward-shaping gap (sparse goal-differential never incentivized
conditioning on time/score, despite the information being available),
not an architectural blindness gap (Pardo et al. state-aliasing argument).
Does not invalidate any built component — the 3.4.3 zero-padding mechanism
still functions the same way mechanically either way — but Section 3.1's
motivating paragraph and Section 3.4.1's "time-blind critic" claim need
rewriting to match. Also affects the actor/critic wrapper design: context
injection must be built against the real 192-dim shape, not 115.
**Queued for the full methodology rewrite pass** (post-ablation-start, per
explicit decision — corrections batched together rather than made
piecemeal mid-build), alongside: T_norm direction (3.1/3.2.2), 137-dim
itemization (3.3.1), CSI `||`→signed notation.

## 17. Decision point — stopped here for tonight
Confirmed: no literal feature duplication (steps_left/scores feed only the
context slot, nowhere else in the 137 dims). Checkpoint from step 15/16
explicitly marked: **exists, does not pass the gate, usable for pilot
throughput/pipeline-wiring purposes only — not validated for real ablation
training.** Remedy options identified but not yet built: (a) extend training
steps and see if the FiLM trend continues, (b) add an auxiliary
context-prediction loss head to force representational pressure toward
context-awareness. Neither pursued yet — deferred to next session, since
throughput measurement doesn't require a gate-passing discriminator.


Entry 1  — Sprint action index: assumed vs verified

Problem: Initial metric implementation used ACTION_SPRINT = 12 for computing Sprint Action Percentage (SAP), based on a commonly-cited GRF action ordering. Results looked plausible on first inspection (no errors, values in a believable range), which risked the bug going undetected.

Hypothesis: GRF's 19-action discrete space follows the documented ordering where action index 12 corresponds to shot, not sprint (index 13). Additionally, sprint may function as a "sticky action" (a toggled, persistent state) rather than a discrete per-step action, meaning even the correct index could not be read directly from the action array.

Diagnostic method: Built an isolated verification script (inspect_sticky_actions.py) that takes a single controlled agent through a scripted sequence — SPRINT, then IDLE (to test persistence), then RELEASE_SPRINT, then DRIBBLE, then RELEASE_DRIBBLE — while printing the full sticky_actions vector after each step. This produces an unambiguous, empirical answer rather than relying on external documentation.

Resolution: Confirmed sticky_actions[8] is the sprint bit — it flips 0→1 on SPRINT, persists through an intervening IDLE action (confirming sticky/toggle semantics), and flips back 1→0 on RELEASE_SPRINT. This also matched the framework's own source code (enhanced_LightActionMask_5.py, is_sprinting = obs["sticky_actions"][8]), providing independent triangulating confirmation. SAP and SBF metric functions were rewritten to consume the sticky-action time series rather than the discrete action array.

What this demonstrates (for Implementation chapter): Rather than trusting either commonly-cited documentation or a plausible-looking first result, an isolated empirical test was designed specifically to disambiguate index and semantics before the metric was trusted for any downstream analysis. The subsequent cross-confirmation against the framework's own source code strengthens confidence that the metric is measuring what it claims to measure.

Marking criteria link: Criterion 5 (technical competence, independent verification rather than assumption); Criterion 4 (risk — "metric validity" was an unstated but real risk, mitigated through direct empirical testing rather than trusting documentation).

Entry 2  — GRF/gym version incompatibility

Problem: GRF_MARL's requirements.txt specifies gym==0.22.0, but gfootball requires gym<=0.21.0. Installing the stated requirements would silently break the already-verified GRF environment.

Hypothesis: The two packages' version constraints are genuinely incompatible for a direct joint install; installing gym 0.22.0 would either break gfootball immediately (detectable) or introduce a subtle behavioural change (harder to detect, more dangerous).

Diagnostic method: Rather than installing requirements.txt as-is, filtered out the conflicting line, installed everything else, then tested the two gym versions in isolation: (a) confirmed gfootball worked correctly under the existing gym==0.21.0, (b) deliberately installed gym==0.22.0 and re-ran the exact same environment-creation and episode-stepping test to see whether it broke, (c) reverted to 0.21.0 and re-ran a 500-step regression test comparing measured Sprint Action Percentage against the theoretically-predicted value for a uniform-random policy.

Resolution: Kept gym==0.21.0, excluded the conflicting pin from the filtered requirements install. Verified no regression via the theoretical-value comparison (see Entry 3).

What this demonstrates (for Implementation chapter): Dependency conflicts between a general-purpose framework's requirements and a narrower environment's requirements were resolved through direct empirical A/B testing rather than trusting either package's stated compatibility range, with a quantitative regression test used to confirm the resolution introduced no silent behavioural change.

Marking criteria link: Criterion 4 (risk identification and mitigation — dependency/environment risk explicitly tested, not assumed); Criterion 5 (dealing with problems appropriately).

Entry 3  — Closed-form statistical validation of metric pipeline

Problem: After building the SAP/SBF metric computation pipeline, needed a way to confirm the entire pipeline (environment stepping, sticky-action extraction, metric calculation) was correct end-to-end, beyond just "the code runs without error."

Hypothesis: For a uniform-random policy sampling independently from 19 discrete actions each step, the sprint sticky bit forms a simple two-state Markov chain with symmetric transition probability 1/19 in each direction (since exactly one action turns sprint on, one turns it off, and all others are no-ops with respect to this state). The stationary distribution of a symmetric two-state chain is exactly 50/50, giving a closed-form theoretical prediction: SAP ≈ 50%, and expected sprint-bout transitions over N steps ≈ N/19.

Diagnostic method: Ran 500 steps of a random policy, computed empirical SAP and SBF, compared directly against the closed-form predictions (50% and 3000/19 ≈ 157.9 respectively).

Resolution: Measured SAP = 49.88% ± 1.89% (500 episodes) and SBF = 157.99 ± 6.0 — both matching the theoretical prediction to within sampling noise. This provided strong, independently-derived confirmation that the metric pipeline (not just the sprint index, but the full chain: environment stepping, observation extraction, sticky-action logging, and the SAP/SBF formulas themselves) is correct, rather than relying solely on unit tests against synthetic data.

What this demonstrates (for Implementation chapter): Where possible, empirical results were validated against independently-derived theoretical predictions rather than relying only on internal consistency checks (unit tests), providing a stronger form of correctness evidence for the core measurement infrastructure the rest of the dissertation depends on.

Marking criteria link: Criterion 5 (exceptional technical rigour — this is a genuinely strong piece of evidence for the 80+ band specifically, since it goes beyond standard software testing practice into statistical/theoretical validation).

Entry 4  — PartialLayernorm dimension mismatch across checkpoints

Problem: A multi-checkpoint comparison script, written and validated against one checkpoint (PassingMain_v2, 192-dim input), crashed on parallel execution with RuntimeError: Given normalized_shape=[133], expected input with shape [*, 133], but got input of size[1, 192] for several (but not all) of the other candidate checkpoints.

Hypothesis: Not all checkpoints in the candidate pool necessarily share the same architecture; some may use an older or simpler encoder (133-dim) rather than the enhanced 192-dim encoder verified for PassingMain_v2. The assumption that "same repository, same task, same architecture" had not actually been verified across the full candidate set.

Diagnostic method: Wrote a small diagnostic script reading each candidate checkpoint's desc.pkl metadata directly, extracting the declared model name and observation space shape for all nine candidates before attempting any further inference.

Resolution: Confirmed six of nine candidates used the verified 192-dim enhanced_LightActionMask_5 architecture; three (3-1_LongPass, 3-1_formation, defense_v3) used a different, undeclared, 133-dim architecture. Given time constraints, scope was deliberately restricted to the six architecturally-consistent candidates rather than building support for a second architecture, with this exclusion explicitly documented.

What this demonstrates (for Implementation chapter): An implicit assumption (architectural consistency across a checkpoint population) was not verified before use, was caught via a crash during a batch operation rather than a single-instance test, and was resolved by adding an explicit pre-flight verification step rather than patching the symptom. The scope decision to exclude incompatible checkpoints was made deliberately and documented, rather than either being ignored or consuming disproportionate time to support a second architecture.

Marking criteria link: Criterion 4 (methodology — three-part justification: choice, alternative considered, limitation acknowledged); Criterion 5 (problem-solving, appropriate scoping of effort under time constraint).

Entry 5  — Multiprocessing OOM and CPU oversubscription

Problem: A 6-worker parallel checkpoint comparison script silently hung for over 30 minutes with no progress output, was suspected stuck, and a second instance was launched in a separate terminal without realising the first was still alive — resulting in 12 concurrent heavy processes and an eventual OOM (out-of-memory) kill of the second instance.

Hypothesis: Two compounding issues: (1) PyTorch was not restricted to a single thread per worker process, causing each of the 6 "workers" to actually consume ~2.2 CPU cores each via internal multi-threading, leading to severe CPU oversubscription; (2) the script provided no live progress feedback, making a slow-but-healthy run indistinguishable from a genuinely hung one.

Diagnostic method: Used ps aux to inspect actual process state and accumulated CPU time (not just wall-clock elapsed time), revealing both processes were alive and consuming substantial CPU (confirming "slow," not "stuck") and that CPU% per worker was ~220%, revealing the oversubscription. Checked dmesg/journalctl patterns for OOM-killer signatures to confirm the second run's "Killed" message was memory-related, not a code exception.

Resolution: Rebuilt the parallel execution architecture: (1) torch.set_num_threads(1) pinned at the start of every worker process, (2) switched from "one process per checkpoint" to a bounded multiprocessing.Pool with an explicit worker cap, (3) added live per-episode progress reporting via a shared multiprocessing.Manager().Queue(), removing the ambiguity that caused the duplicate-launch mistake in the first place.

What this demonstrates (for Implementation chapter): A production-relevant infrastructure failure mode (oversubscription under naive multiprocessing) was diagnosed through direct OS-level process inspection rather than guesswork, and the fix addressed the root architectural cause (unbounded thread proliferation, lack of progress visibility) rather than just increasing available memory or reducing episode counts as a workaround. This directly informed the design of the subsequent Blackwell-scale (20-worker) production run.

Marking criteria link: Criterion 4 (risk register — "compute failure" risk, mitigation designed and then genuinely validated under a real failure); Criterion 5 (systematic debugging under a production-adjacent failure, not a toy example).

Entry 6  — pandas/numpy binary incompatibility after environment churn

Problem: After a sequence of package installs to resolve build-isolation issues (psutil, six, setuptools pinning) during gfootball/gym installation on the second (Blackwell) machine, an unrelated script failed with ValueError: numpy.dtype size changed, may indicate binary incompatibility.

Hypothesis: One of the incidental package installs silently upgraded either numpy or pandas to a version incompatible with the other's compiled C extension ABI, without either package being deliberately targeted.

Diagnostic method: Isolated the two packages' import behaviour independently (numpy imported cleanly; pandas failed specifically inside its compiled _libs extension), confirming the issue was a compiled-binary mismatch rather than a pure-Python logic error.

Resolution: Force-reinstalled both packages together at the exact version pair already confirmed working on the first machine (numpy==1.23.5, pandas==1.5.3), re-verified both imported cleanly, then re-ran the GRF environment regression test as an additional safety check before proceeding.

What this demonstrates (for Implementation chapter): Cross-machine environment replication is not guaranteed to be identical even when following the same nominal setup steps, particularly when incidental dependency resolution occurs; this was caught by testing rather than assumed, and resolved by pinning to a previously-validated known-good version pair rather than trial-and-error version guessing.

Marking criteria link: Criterion 4 (reproducibility as a risk category, explicit mitigation); Criterion 5 (systematic root-cause diagnosis).


# Debugging Log — New Entries, This Session

Numbered independently starting at #1 — renumber to continue from the
project's actual running count (last known reference: entries up to at
least #20 exist in the main log, region-label inversion). These are NOT
final numbers.

---

### Entry #1 — Blackwell path regression via git pull
**Symptom:** `ModuleNotFoundError: No module named 'enhanced_LightActionMask_5'`
on Blackwell, running `record_baseline_rollout.py`.
**Root cause:** `PROJECT_ROOT`/`GRF_MARL_ROOT` were hardcoded, edited to the
laptop's `/home/urstr/...` path for a laptop run days earlier, committed and
pushed. A later `git pull` on Blackwell overwrote Blackwell's own correct
`/home/u5749464/...` path with the laptop's. `GRF_MARL_ROOT` pointed at a
nonexistent directory, so the `sys.path.insert` added a dead path — the
"module not found" was a downstream symptom, not the real fault.
**Fix:** Replaced hardcoded paths with
`os.path.expanduser("~/dissertation/...")`, resolves correctly per-machine
via `$HOME`, no more manual edits or collision risk on future pulls.
**Verification:** Re-ran cleanly on Blackwell after the fix.

### Entry #2 — Wrong conda env, missing numpy
**Symptom:** `ModuleNotFoundError: No module named 'numpy'` running
`build_expert_dataset.py`.
**Root cause:** Shell was in `dissertation-train`, not `dissertation-grf` —
all discriminator pipeline work in this session assumed the latter.
**Fix:** `conda activate dissertation-grf`.
**Verification:** Ran cleanly after switching.

### Entry #3 — Wrong working directory for a cross-referenced script
**Symptom:** `agent_cell_histogram.py: No such file or directory`, run from
`src/grf_baseline/`.
**Root cause:** Script lives in `src/discriminator/` and imports
`build_expert_dataset.classify_bin` — needs to run from that directory.
**Fix:** `cd` to `src/discriminator/` before running.
**Verification:** Ran cleanly.

### Entry #4 — Missing guard for the all-cells-empty case
**Symptom:** `ValueError: need at least one array to concatenate` in
`context_balanced_sampler.py`'s `sample()`, under `on_empty='skip'`, when
the ONLY requested cell in a batch was empty (so `idx_parts` ended up
completely empty — not just one empty cell among several).
**Root cause:** The `on_empty='skip'` implementation as specified included
a guard (`if not idx_parts: return empty arrays`) placed before the final
`np.concatenate(idx_parts)` — but the first applied edit omitted it, so
`np.concatenate` was called on an empty list directly.
**Fix:** Added the guard immediately before the concatenate line.
**Verification:** `test_on_empty_handling.py` check 2 (all-cells-empty
batch) passed after the fix; re-ran full test suite, all green.

### Entry #5 — Counterfactual gate FAILED, first architecture (concat)
**Symptom:** Real gate run against the trained (concat) Phase B checkpoint:
mean|shift| ≈ 0.003–0.004 both directions, 0% of examples exceeding the
locked 0.1 threshold. Structurally consistent (nonzero, correctly signed),
just far too small — ruled out a wiring bug immediately.
**Root cause (diagnosed via `diagnose_context_sensitivity.py`, see
Implementation Log #13):** Dilution, not suppression, not a generalization
gap. 135 non-context dims already gave ~96% train+held-out separability —
gradient descent had no incentive to develop sensitivity to 2 additively-
concatenated context dims when the rest of the input already solved the
classification task. Confirmed via: (a) first-layer weight norm for context
mid-pack, not starved; (b) input-gradient for context only ~2x below
average, not order-of-magnitude, inconsistent with R1 actively suppressing
it specifically; (c) gate FAILED even on TRAIN data — the model never
learned this relationship at all, so there was nothing to fail to
generalize.
**Fix:** Rewrote `discriminator_model.py` to use FiLM conditioning
(Section 3.3.1's own pre-registered alternative for this exact failure
mode) instead of input-layer concatenation — context now modulates hidden
activations directly (γ, β per layer) rather than competing as 2 more
input dims among 135.
**Verification:** Re-ran both pre-training phases from scratch (FiLM,
fresh init — old checkpoints architecturally incompatible), re-ran the
real gate.

### Entry #6 — Counterfactual gate STILL FAILED, second architecture (FiLM)
**Symptom:** Real gate run against the trained FiLM Phase B checkpoint:
late_win→early_loss mean|shift| 0.0073 (was 0.0037, ~2x); early_loss→late_win
mean|shift| 0.0523 (was 0.0043, ~12x); 2.8% of examples in the second
direction now exceed 0.1 (was 0%/0%). Genuine, substantial movement — FiLM
demonstrably gave the network capacity to use context — but still short of
the 0.1 mean threshold in both directions.
**Root cause (partial, not fully resolved):** FiLM addressed *capacity*,
not *incentive*. Two contributing factors identified, not yet disentangled:
(1) real correlational signal already available elsewhere — the action
block's sprint-frequency statistics correlate with true context in human
data (per the project's own locked SAP-by-outcome findings), giving the
network an easier path to partial separability that happens to be
context-correlated without literally reading the context dims; (2) a
larger, genuinely context-*independent* shortcut — the frozen baseline's
near-constant ~93.5% SAP vs. demonstrators' lower SAP in essentially every
region, which alone plausibly explains most of the ~96% separability
regardless of context reasoning. Confirmed: no literal feature duplication
(steps_left/score fields feed ONLY the context slot, verified by re-reading
`compute_raw_features` line by line — nothing else touches them).
**Status: UNRESOLVED, deferred.** Two remedies identified, neither
attempted yet: (a) extend training steps, check if the 2x→12x trend
continues; (b) add an auxiliary context-prediction loss head to create
direct representational pressure. Checkpoint from this run is usable for
throughput/pilot purposes only — explicitly NOT validated as a real,
gate-passing discriminator. Do not use it in any ablation condition without
resolving this first.
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
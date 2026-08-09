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
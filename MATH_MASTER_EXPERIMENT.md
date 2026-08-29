# Cumulative 100k master mathematics curriculum experiment

This experiment exercises the complete curriculum mechanism on a local 12GB
GPU without pretending to be the final large mathematics corpus. It contains
exactly 100,000 distinct canonical tower-training objects, plus 516 disjoint
validation and 516 disjoint test objects, spanning KS1 through graduate
mathematics and formal research preparation. Natural-language paraphrases are
retained as dispatcher metadata rather than duplicated tower targets.

The experiment has 15 ordered phases. The first five retain the detailed KS1
progression. Later phases cover KS2, secondary mathematics, GCSE, A-level,
undergraduate calculus/linear algebra/discrete mathematics/probability/algebra,
graduate analysis/algebra, and formal research-preparation tasks.

Every answer and structured derivation is recomputed by the dataset auditor.
The research-preparation phase tests identities, counterexamples and invariants;
it does not claim to train or evaluate novel open-problem research.

Each phase-specific training view contains its active shard plus
criterion-balanced replay from every earlier phase. Future-phase examples are
present in separately sealed files but never appear in an active training view.
Validation and test use unseen mathematical objects within each criterion's
declared taught domain.

Build and audit locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_math_master_experiment_v1.ps1
```

The output is immutable. Set `-OutputRoot` to a new directory for another
revision. The lightweight 12GB training contract uses small batches, mixed
precision and phase-gated resume.

Start or exactly resume the local CUDA experiment with one command:

```powershell
python -m tools.run_math_master_experiment auto
```

Each phase has a competency-gated budget of 10–60 epochs. It may advance
after the minimum only when every active and cumulative-retention gate passes
twice consecutively. Each epoch samples 512 objects: 100% active in phase 1,
then 75% active and 25% criterion-balanced cumulative replay. This provides up
to 3,840 optimizer steps per phase at batch size 8 while retaining early
promotion when a phase is mastered sooner.

The experiment writes under
`C:\CFTN\artifacts\math_master_experiment_100k_v1`. Earlier small-corpus and
failed-build artifacts remain preserved as evidence.

The local read-only dashboard is available on the LAN at
`http://192.168.1.128:8789/`. It refreshes every 30 seconds and reports the
current 10–60 epoch phase budget, gate streak, validation trend, curriculum
schedule, and recent checkpoints in a single-column layout.

Before the first full run, a one-batch CUDA plumbing check is available:

```powershell
python -m tools.run_math_master_experiment smoke
```

The runner builds and audits missing data, uses an 8-layer 256-wide tower with
batch size 8 and BF16, and resumes automatically when its artifact already has
a compatible checkpoint. It stops fail-closed at the first phase that reaches
its cap without two consecutive complete acceptance passes.

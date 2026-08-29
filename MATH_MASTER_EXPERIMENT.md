# Compact master mathematics curriculum experiment

This experiment exercises the complete curriculum mechanism on a local 12GB
GPU without pretending to be the final large mathematics corpus. It contains a
small representative shard at every level from KS1 through graduate mathematics
and formal research preparation.

The experiment has 15 ordered phases. The first five retain the detailed KS1
progression. Later phases cover KS2, secondary mathematics, GCSE, A-level,
undergraduate calculus/linear algebra/discrete mathematics/probability/algebra,
graduate analysis/algebra, and formal research-preparation tasks.

Every answer and structured derivation is recomputed by the dataset auditor.
The research-preparation phase tests identities, counterexamples and invariants;
it does not claim to train or evaluate novel open-problem research.

Each phase-specific training view contains all of its small active shard plus
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

Before the first full run, a one-batch CUDA plumbing check is available:

```powershell
python -m tools.run_math_master_experiment smoke
```

The runner builds and audits missing data, uses an 8-layer 256-wide tower with
batch size 8 and BF16, and resumes automatically when its artifact already has
a compatible checkpoint. Each phase has at most eight epochs and needs two
consecutive complete acceptance passes. It stops fail-closed at the first phase
that reaches its cap without passing.

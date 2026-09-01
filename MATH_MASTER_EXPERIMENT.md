# Cumulative 100k master mathematics curriculum experiment

The current `v4` recipe uses canonical `add` math IR for KS1 part-whole
composition, compact executable count-on traces, result-balanced active/replay
sampling for criterion `1AS-1`, and result-stratified held-out examples. The
completed `v3` dataset and artifacts remain immutable comparison evidence.

This experiment exercises the complete curriculum mechanism on a local 12GB
GPU without pretending to be the final large mathematics corpus. It contains
exactly 100,000 distinct canonical tower-training objects, plus 526 disjoint
validation and 526 disjoint test objects, spanning KS1 through graduate
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

Sampling is hierarchical: criteria are balanced first, then operations within
each criterion. Every phase starts a fresh optimizer and a phase-local cosine
schedule with a three-epoch warmup from zero to `3e-4`, decaying to `3e-5` by
epoch 60. Model weights carry forward; optimizer momentum does not. Active
validation is deterministically stratified by operation as well as criterion.

The experiment writes under
`C:\CFTN\artifacts\math_master_experiment_100k_v3`. Earlier small-corpus and
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

## V10 reproducibility record: Stage 4 multiplication promotion

V10 is a sealed successor experiment for the Stage 4 multiplication failure.
It does not alter earlier experiment data or the phase-gate thresholds. Its
dataset configuration is `config/math_master_experiment_v10.json`, with:

- recipe: `canonical_v10_multiplication_decomposition_v1`;
- generator version: `v10_multiplication_decomposition_v1`;
- 110,392 training records and semantic train/validation/test disjointness;
- only the original Stage 4 multiplication criterion (`2MD-1`) expanded to
  8,000 records; and
- 75% active / 25% cumulative criterion-balanced replay, with future-phase
  examples forbidden from active training exposure.

The sealed manifest is
`C:/CFTN/.datasets/math_master_experiment_v10/manifest.json` (audit: passed;
manifest SHA-256 `d41441fbe3d6c80d2c714f808b06e6ba6da1e30eb8d676babfdc02f7b119cd1c`;
generator SHA-256 `714ec9cf0fef57b0242b07d54be1f0a8f3f50ba73ab617de2e157f9374f53704`).

The complete implementation lineage is retained on
`codex/v2-math-stage2-recovery`:

- `056556d` — versioned V10 generator, V10 dataset config, local config, and
  bilateral multiplication decomposition;
- `15fa8c7` — V10 dashboard contract support; and
- `5637013` — Stage 4's 512-token evaluation allowance.

The corresponding source/configuration surface is
`cftn_text/math_curriculum_data.py`, `tools/run_math_master_experiment.py`,
`tools/serve_math_master_dashboard.py`,
`config/math_master_experiment_v10.json`, and
`config/math_master_experiment_local_v10.yaml`. This list, the sealed
manifest hashes above, and the durable run artifact together are the V10
reproduction record; the generated data and checkpoints remain outside Git as
immutable run evidence.

V10 structured derivations grow with the taught operation. Every V10
generation-validation panel therefore has a 2,048-token allowance; the former
224-token default truncated valid traces before their answer tag in both Stage
4 and Stage 5. This changes evaluation capacity only, not the model, data,
optimizer, curriculum, or acceptance gates. A resume-path repair also restores
the phase-local schedule reporting fields from the saved curriculum state,
without changing the restored optimizer or scheduler state.

Run or resume V10 exactly with:

```powershell
C:\Users\adria\anaconda3\python.exe -u -m tools.run_math_master_experiment auto `
  --config config/math_master_experiment_local_v10.yaml `
  --dataset-config config/math_master_experiment_v10.json `
  --data C:\CFTN\.datasets\math_master_experiment_v10 `
  --artifact C:\CFTN\artifacts\math_master_experiment_v10\run_r2 `
  --device cuda --contract-profile v10_multiplication
```

Promotion evidence is retained in
`C:/CFTN/artifacts/math_master_experiment_v10/run_r2/metrics.jsonl`. In the
fixed Stage 4 active panel, epoch 98 passed at 95.83% answer accuracy, 91.67%
multiplication accuracy, and 100% valid answers. Epoch 99 supplied the second
consecutive complete pass: 87.50% answer accuracy, 75.00% multiplication,
100% division, and 100% valid answers. The curriculum therefore advanced to
`ks2_four_operations` (Stage 5). This is an intermediate promotion record, not
a claim that the full V10 curriculum has completed.

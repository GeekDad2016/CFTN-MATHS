# CFTN-Text V1.1 canonical run review

Last evidence snapshot: **2026-08-09 12:37:40 Europe/London**

Run state at snapshot: **Stage 8 of 15 running**
Scope: **V1.1 algorithmic one-variable linear-equation experiment only**

This is the canonical interpretation record for the run under
`G:/ctfn-text/artifacts/v1_1_algorithmic_linear_equations`. Future reviews must
read this file before interpreting later reports. Do not combine these findings
with the earlier V1 experiment, the visual CFTN experiments, or a future V2 run.

## Executive verdict

The evidence supports three different conclusions that must not be conflated:

1. **The standalone math tower learned the familiar training distribution.** It
   reached 99.98% exact greedy-generation accuracy on the familiar sealed test.
2. **The math-to-GPT bridge works causally.** Correct specialist messages let the
   frozen GPT tower emit the specialist answer; disabling or shuffling those
   messages destroys the effect on familiar and input-extrapolation examples.
3. **The current bidirectional model has not demonstrated collaboration or
   generalization.** In shared-view Stage 8, GPT-to-math communication damages
   the math tower, the learned gate does not suppress the damage, and the full
   CFTN is worse than the standalone math tower on every completed split.

Therefore this run has produced a meaningful partial result, not a CFTN success:

> A frozen specialist can causally inject an answer-specific representation into
> a frozen GPT tower through a compact learned bridge.

It has **not** yet shown that useful information flows in both directions, that
the towers solve tasks synergistically, or that the system generalizes.

## Immutable run identity

| Item | Value |
|---|---|
| Project | `cftn_text_v1_1_algorithmic_linear_equations` |
| Seed | `719` |
| Config | `G:/ctfn-text/config/v1_1_algorithmic_linear_equations.yaml` |
| Canonical parsed-config SHA-256 | `6beeecb14596720700d1e4865f996116896753ee37366f2a754e3e95b15d37c6` |
| Data manifest SHA-256 | `66d5243c6af2f33e745d3cfc5eeb86b90b3c735c41cafe120627b3d14e1a8e33` |
| Synergy protocol SHA-256 | `8e7331263a9ec4334016ba1950bf630d8aadfe9bc6fc91ce94982d181b755187` |
| Synergy manifest SHA-256 | `0c290f0b5e82658e7ff2e83e8ce0896b6a1b77bb6732f3bbdfa559b5a60d1324` |
| Frozen GPT | Hugging Face `gpt2`, commit `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Math checkpoint | `math/math.best.pth`, SHA-256 `97a48a484157f10e9ce41a8d9fef8ea8b168180d26ebee00d50f549bc506eb27` |
| M2G checkpoint | `bridge_m2g_contextual/bridge_m2g.best.pth`, SHA-256 `fd3aaf73c710fcb30544960af2dfa928c3a1d1b6cecec46c2daa721574574ef3` |
| Bidirectional checkpoint used by Stage 8 | `bridge_bidirectional_contextual_complementary/bridge_bidirectional.best.pth`, SHA-256 `97884cf1697ca97f2d4c2ffcb569b0d2e993a883c1ace51910ff487ffb99d85b` |

Any result associated with a different config, manifest, seed, or checkpoint hash
is a different experiment and must be reported separately.

The config hash above is calculated from canonical parsed YAML, not from the raw
file bytes; comments and formatting therefore do not affect the run identity.

## Data and model contract

The generated corpus contains **165,000 unique records**:

| Split | Records | Purpose |
|---|---:|---|
| Calibration | 5,000 | Frozen GPT capability check |
| Train | 100,000 | Familiar templates and four numeric curriculum bands |
| Validation | 10,000 | Checkpoint selection; familiar support |
| Test | 10,000 | Sealed familiar-distribution generation |
| Held-out language | 10,000 | Four unseen linguistic templates |
| Input extrapolation | 10,000 | Coefficients/intercepts beyond training support |
| Answer extrapolation | 10,000 | Answers with absolute value 201-400 |
| Compositional | 10,000 | Four unseen composed equation templates |

The manifest reports zero normalized-equation overlap between splits. Training
used a 100-epoch curriculum with 100,000 sampled examples per epoch, but the
unique training corpus remained 100,000 records.

The sealed causal benchmark contains **5,000 counterfactual pairs / 10,000
rows**, with 1,000 pairs from each of the five evaluation splits and zero
counterfactual-source overlap.

Model details:

- GPT-2 remains fully frozen.
- The byte-level math transformer has 8 layers, hidden size 384, 6 attention
  heads, feed-forward size 1,536, and 14,789,249 trainable parameters during its
  standalone training.
- The answer classifier is disabled; success requires generated signed digits.
- Each bridge sends 8 message tokens of width 256 through contextual gates.
- M2G training exposes 2,935,556 trainable bridge, gate, and GPT receiver-adapter
  parameters.
- Bidirectional training exposes 4,783,623 trainable bridge, gate, and receiver-
  adapter parameters.
- The pretrained GPT weights and core math-transformer weights remain frozen
  during bridge training; the attached receiver adapters are trainable.

## Stage-by-stage evidence

### Stages 1-2: data and causal benchmark preparation — passed integrity checks

- All requested splits were generated.
- Data and benchmark manifests were hashed.
- Normalized equation overlap is zero.
- Counterfactual source overlap is zero.
- These are integrity passes, not model-capability passes.

### Stage 3: frozen GPT baseline — calibration decision passed, capability failed

On 5,000 calibration examples:

| Frozen GPT-2 measure | Result |
|---|---:|
| Zero-shot strict exact accuracy | 0.00% |
| Zero-shot lenient accuracy | 0.02% |
| Three-shot strict/lenient accuracy | 0.68% |
| Eight-candidate ranking accuracy | 1.30% |
| Random candidate-ranking chance | 12.50% |
| Prompt-copy rate | 25.60% |

The calibration decision correctly allowed the experiment to continue because
GPT-2 was weak enough to leave 98.7 percentage points of measured headroom. This
was **not** a claim that GPT-2 understood or solved the mathematics.

### Stage 4: standalone math-tower training — completed successfully

- Ran all 100 configured epochs; best checkpoint was epoch 98.
- Training time: 14,491 seconds (about 4 hours 2 minutes).
- Best familiar validation teacher-forced sequence accuracy: 99.96%.
- Best familiar validation token accuracy: 99.9994%.
- Best validation loss: approximately `4.13e-5`.
- No answer-head shortcut was active.

Selected learning-curve snapshots:

| Epoch | Curriculum phase | Validation sequence | Validation token | Validation loss |
|---:|---|---:|---:|---:|
| 1 | foundations | 0.22% | 86.01% | 0.7572 |
| 10 | foundations | 15.01% | 87.35% | 1.4954 |
| 30 | two-digit | 41.66% | 93.90% | 0.7517 |
| 60 | three-digit | 69.09% | 98.34% | 0.1879 |
| 70 | full support | 99.82% | 99.9975% | 0.000116 |
| 98 | full support | 99.96% | 99.9994% | 0.0000413 |

This proves strong fitting of familiar validation sequences. Teacher forcing is
not evidence of robust autoregressive or out-of-distribution generation.

### Stage 5: standalone math-tower greedy generation — familiar pass, generalization failure

Each split contains 10,000 sealed examples:

| Split | Exact accuracy | Valid-answer rate | Canonical trace exact | Verdict |
|---|---:|---:|---:|---|
| Familiar test | **99.98%** | 100.00% | 99.98% | Pass |
| Held-out language | 0.80% | 86.45% | 0.03% | Fail |
| Input extrapolation | 20.87% | 100.00% | 4.08% | Fail |
| Answer extrapolation | 0.67% | 100.00% | 0.66% | Fail |
| Compositional | 0.65% | 99.96% | 0.14% | Fail |

The overall specialist acceptance gate failed. The tower learned familiar
templates and numeric support but did not learn a broadly general algorithm for
unseen wording, new compositions, or larger answer ranges.

### Stage 6: contextual math-to-GPT bridge — training objective passed

- Shared-view M2G-only training stopped at epoch 63 after a validation plateau.
- Best checkpoint: epoch 48.
- Training time: 54,877 seconds (about 15 hours 15 minutes).
- Best teacher-forced GPT sequence/token accuracy: 100% / 100%.
- Math sequence/token accuracy remained 99.96% / 99.9994%.
- Best shuffled-loss gap: 5.0437.
- Mean M2G sender gate at the best checkpoint: 0.3377.
- Collapse guard never triggered.

These training metrics showed strong message dependence. Stage 8 later supplied
the stronger autoregressive causal evidence for this direction.

### Stage 7: contextual bidirectional bridge — training objective passed, generation claim unresolved

This stage trained on **complementary private views**, not ordinary shared
prompts. GPT received role semantics without the true numbers; the math tower
received numeric slots without role meanings. Consequently every training
example required communication.

- Training recovered from weak early math output without a restart.
- It stopped at epoch 67 after a validation plateau.
- The selected best checkpoint was epoch 52, not the final epoch.
- Training time: 73,504 seconds (about 20 hours 25 minutes).
- Effective bridge LR was capped at `5e-5`; contextual-gate LR used a 0.5
  multiplier. The configured `2e-4` request was not used directly.
- Collapse guard never triggered.

Key teacher-forced validation trajectory:

| Epoch | GPT sequence | Math sequence | Math token | Loss | Shuffled gap | Mean G2M gate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 0.00% | 85.98% | 0.4437 | 4.9428 | 0.5743 |
| 12 | 100.00% | 0.05% | 93.55% | 0.1971 | 5.1123 | 0.9770 |
| 20 | 100.00% | 61.21% | 99.26% | 0.02358 | 5.5531 | 0.9299 |
| 30 | 100.00% | 95.37% | 99.9349% | 0.002242 | 5.8788 | 0.4750 |
| 52, selected best | 100.00% | 98.79% | 99.9828% | 0.000704 | **6.0257** | 0.4945 |
| 67, final | 100.00% | 99.21% | 99.9883% | 0.000484 | 5.7359 | 0.4775 |

The selected checkpoint maximized the preregistered balanced selection metric,
which rewarded sequence performance and shuffled-message dependence. The very
strong teacher-forced numbers did **not** guarantee greedy generation quality.

### Stage 8: shared-view causal evaluation — running, but already fails the configured gate

Stage 8 uses the epoch-52 bidirectional best checkpoint. Both towers receive the
ordinary complete problem, then seven arms test correct, disabled, and shuffled
communication. The first three 10,000-example splits are final:

| Completed split | Full CFTN GPT | Full CFTN math | GPT alone | Math alone | One-way M2G final GPT | Both shuffled GPT |
|---|---:|---:|---:|---:|---:|---:|
| Familiar test | 79.69% | 79.69% | 0.00% | **99.98%** | **99.98%** | 0.40% |
| Held-out language | 0.68% | 0.63% | 0.00% | **0.80%** | **0.85%** | 0.54% |
| Input extrapolation | 12.81% | 12.81% | 0.00% | **20.87%** | **20.87%** | 0.31% |

`One-way M2G` means GPT-to-math is disabled while math-to-GPT remains active.
It is not the same as GPT alone.

#### Causal effects in completed Stage 8 splits

| Effect in percentage points | Familiar | Held-out language | Input extrapolation |
|---|---:|---:|---:|
| Full CFTN minus strongest individual (synergy) | **-20.29** | **-0.12** | **-8.06** |
| G2M direct effect at math readout | **-20.29** | **-0.17** | **-8.06** |
| Correct G2M minus shuffled G2M | +0.31 | -0.10 | -0.37 |
| M2G direct effect at GPT readout | +79.69 | +0.68 | +12.81 |
| Correct M2G minus shuffled M2G | **+79.26** | +0.02 | **+12.49** |
| Full CFTN minus both-shuffled GPT | **+79.29** | +0.14 | **+12.50** |

The correct G2M message performs essentially like a shuffled G2M message. Its
content is not providing useful example-specific information in shared view;
the injected representation mostly acts as a harmful perturbation.

The M2G result is different. On familiar and input-extrapolation examples,
disabling or shuffling M2G removes almost all GPT accuracy. With G2M disabled,
the final GPT prediction matches the math prediction on 100.00% of familiar
examples and 99.82% of input-extrapolation examples. This is the strongest
positive result in the run so far.

#### Rescue-versus-harm accounting

This compares full-CFTN GPT correctness against standalone-math correctness:

| Split | Both correct | CFTN rescue | CFTN harm | Both wrong |
|---|---:|---:|---:|---:|
| Familiar | 7,969 | **0** | **2,029** | 2 |
| Held-out language | 13 | 55 | 67 | 9,865 |
| Input extrapolation | 1,048 | 233 | **1,039** | 7,680 |

The apparent rescues are not proof that GPT semantics helped mathematics,
because shuffled G2M performs equally well or better overall. They are currently
consistent with stochastic-looking representation perturbations occasionally
moving a wrong decode onto the right answer.

#### Gate behavior

| Split | Mean G2M sender gate | Mean M2G sender gate | Mean G2M message norm | Mean M2G message norm |
|---|---:|---:|---:|---:|
| Familiar | 0.2831 | 0.2867 | 7.19 | 4.41 |
| Held-out language | 0.2605 | 0.2708 | 6.59 | 4.18 |
| Input extrapolation | 0.2903 | 0.2958 | 7.38 | 4.54 |

Gates are neither all-zero nor numerically saturated, so this is not classic
gate collapse. However, G2M gate means are almost unchanged between correct and
wrong examples. The gate does not recognize when GPT input is redundant or
harmful and does not protect the specialist.

#### Live Stage 8 snapshot (not a final result)

At 2026-08-09 12:37:40 Europe/London:

- Overall Stage 8 progress: 63.488%.
- Current split: answer extrapolation, 1,744/10,000 examples (17.44%).
- Provisional full-CFTN GPT exact accuracy: 0.00%.
- Provisional full-CFTN math exact accuracy: 0.459%.
- Provisional standalone-math exact accuracy: 0.745%.
- Elapsed Stage 8 time: about 9 hours 49 minutes; ETA about 5 hours 39 minutes.
- Pipeline PID 8656 and evaluator PID 8552 were alive; the RTX 4070 was healthy.

The answer-extrapolation values are provisional. The three earlier split values
are final because all 10,000 examples in each split completed.

The configured Stage 8 requirements are 99% familiar, 90% held-out language,
95% input extrapolation, 80% answer extrapolation, and 90% compositional exact
GPT accuracy. The first three completed criteria have already failed, so the
overall shared-view collaboration gate cannot pass this run.

## What the run has and has not proved

### Supported by current evidence

- The data/benchmark separation and counterfactual construction are intact.
- The small math transformer has enough capacity to solve the familiar task.
- M2G bridge messages contain answer-specific information.
- Frozen GPT-2 can consume that latent message and emit the specialist answer.
- The one-way M2G system preserves the specialist's familiar and extrapolation
  accuracy almost exactly.
- Neither bridge-training run suffered numerical collapse.

### Not supported by current evidence

- Broad algorithmic generalization by the math tower.
- Robust parsing of unseen linguistic formulations.
- Useful, content-specific GPT-to-math communication in shared view.
- Positive synergy over the strongest individual tower.
- A gate that conditionally suppresses redundant or harmful communication.
- Bidirectional CFTN operation as one cooperating system.
- Superiority to a simple specialist-to-GPT serial pipeline.

The current best observed architecture is therefore the one-way path:

```text
complete prompt -> math tower -> M2G bridge -> frozen GPT output
```

with GPT-to-math disabled. This is a useful baseline, not the final CFTN goal.

## Why Stage 7 looked excellent while Stage 8 failed

This difference is expected once the metrics are separated correctly:

1. Stage 7 used teacher forcing, so each next token received the correct prefix.
   Stage 8 used greedy autoregressive generation, where one mistake changes all
   later states.
2. Stage 7 trained complementary views where GPT information was always needed.
   Stage 8 supplied complete prompts to both towers, making G2M redundant.
3. Stage 7 never taught the contextual gate a negative case in which it should
   protect an already-correct math tower by closing or becoming residual-neutral.
4. A large shuffled-loss gap proves dependence on some bridge message during
   training; it does not by itself prove that each direction improves generated
   answers under a different view mode.

Future reviews must always report teacher-forced validation separately from
greedy-generation exact accuracy.

## Unresolved test and decision rule

Stage 9 is the decisive **complementary-view causal synergy evaluation**. It asks
whether GPT-to-math becomes useful when the math tower genuinely lacks role
semantics and GPT lacks the true numbers. Stage 8 does not answer that question.

Let the current sealed evaluation finish unchanged. Do not tune this checkpoint
from Stage 8 test results. After Stage 9:

- If correct G2M beats both disabled and shuffled G2M by at least 2 points with
  a positive 95% interval, and full CFTN beats the strongest individual by at
  least 10 points, the bidirectional hypothesis remains viable despite its
  shared-view safety failure.
- If Stage 9 also lacks content-specific G2M benefit, classify V1.1 as a
  one-direction communication success and a bidirectional-synergy failure.

If redesign is required, preserve the successful M2G path and add all of the
following using training/validation data only:

1. generation-based checkpoint selection rather than teacher forcing alone;
2. a specialist-preservation or distillation loss comparing G2M-on against the
   frozen math baseline;
3. mixed bridge-needed and bridge-unneeded examples, plus bridge dropout;
4. a residual-neutral G2M initialization and an explicit ability to close;
5. stronger algorithmic, paraphrase, compositional, and digit-length curricula;
6. one-way M2G as a mandatory baseline for every future bidirectional run.

## Authoritative artifacts and dashboards

- GPT baseline: `G:/ctfn-text/artifacts/v1_1_algorithmic_linear_equations/gpt_calibration/report.json`
- Math training: [W&B run 33i0qe5o](https://wandb.ai/kaipo/cftn-text/runs/33i0qe5o)
- Math generation report: `G:/ctfn-text/artifacts/v1_1_algorithmic_linear_equations/evaluation_math/report.json`
- M2G training: [W&B run u2wskvvf](https://wandb.ai/kaipo/cftn-text/runs/u2wskvvf)
- Bidirectional training: [W&B run 0is2pwz8](https://wandb.ai/kaipo/cftn-text/runs/0is2pwz8)
- Stage 8 evaluation: [W&B run shve2799](https://wandb.ai/kaipo/cftn-text/runs/shve2799)
- Stage 8 live status: `G:/ctfn-text/artifacts/v1_1_algorithmic_linear_equations/evaluation_bidirectional_contextual_shared/status.json`
- Pipeline status: `G:/ctfn-text/artifacts/v1_1_algorithmic_linear_equations/pipeline_status.json`

When Stage 8 or later stages complete, append a dated update to this file. Do
not rewrite the completed values above unless the checkpoint/config/hash changes;
instead add a new run identity and treat it as a separate experiment.

## Dated update: 2026-08-10 after contextual Stage 9

This section appends evidence; it does not alter the historical Stage-8
snapshot above. The fixed-open Stage-13 evaluation and final Stage-14/15
architecture comparison were still running when this update was written.

### Stage 8 final interpretation

Shared-view evaluation confirmed a practical safety failure. Both towers saw
the complete prompt, so the specialist did not need GPT-to-math context. The
full bidirectional system underperformed the GPT-to-math-disabled one-way arm:

| Split | Full CFTN GPT | Math-side readout with GPT-to-math disabled |
|---|---:|---:|
| Familiar test | 79.69% | 99.98% |
| Held-out language | 0.68% | 0.80% |
| Numerical extrapolation | 12.81% | 20.87% |
| Answer extrapolation | 0.00% | 0.67% |
| Compositional | 0.61% | 0.65% |

Correct and shuffled GPT-to-math messages were nearly indistinguishable in
shared view. V1.1 therefore did not learn to make redundant communication
neutral. This is a no-harm/control failure, not the primary test of the intended
generalist-specialist information division.

### Stage 9 complementary-view result

Stage 9 is the closer match to the intended CFTN mechanism. GPT held operation
and role semantics while the math tower held opaque numeric values. Neither
private view was sufficient alone.

The sealed aggregate result over 10,000 generated examples was:

| Arm | Exact accuracy |
|---|---:|
| Full contextual CFTN | 42.15% |
| Strongest isolated arm | 0.23% |
| Both messages shuffled | 0.33% |
| GPT-to-math disabled math readout | 0.23% |
| Math-to-GPT disabled final GPT readout | 0.00% |

The familiar private-view split reached 98.65% and the compositional split
reached 87.00%. On compositional examples, the correct math readout was 86.80%,
GPT-to-math disabled was 0.45%, and shuffled GPT-to-math was 15.00%. Disabling
or shuffling math-to-GPT likewise removed the final answer. These are strong
causal signs that the frozen towers exchanged indispensable, prompt-specific
information in the setting for which the bridge was trained.

Five causal subgates passed with positive intervals: synergy, correct versus
both shuffled, GPT-to-math gain, math-to-GPT gain, and no regression versus the
serial math readout. The aggregate formal gate still failed because broad OOD
capability was insufficient:

- counterfactual both-correct rate: 38.72% versus 90% required;
- math-to-GPT donor-follow rate: 42.15% versus 80% required.

Generalization remained weak: held-out language was 13.55%, numerical
extrapolation 10.25%, and unseen answer magnitude 1.30%. On held-out language,
correct GPT-to-math exceeded shuffled GPT-to-math by only 1.15 points, so unseen
wording did not produce a robustly content-specific message.

The correct scoped conclusion is therefore:

> V1.1 demonstrated mechanism-level bidirectional cooperation on familiar and
> compositional complementary tasks. It did not demonstrate broad specialist
> generalization, shared-view communication safety, or a mature conditional
> multi-specialist runtime.

### Fixed-open control observed before final comparison

The one-way fixed-open control saved its best checkpoint at epoch 3 and
collapsed at epoch 4. The bidirectional fixed-open control saved its best at
epoch 1 and collapsed at epoch 3. In both cases the shuffled-loss dependence
and generated sequence path deteriorated abruptly. The collapse guards
preserved the earlier checkpoints, which are used by Stage 13. This is evidence
that always-open residual communication is optimization-fragile; Stage 14 remains
the preregistered formal contextual-versus-fixed-open comparison.

### V1.2 decision

Do not attempt to solve the communication defect indirectly by scaling the
specialist first. The immediate revision is a frozen-tower conditional
communication experiment:

1. initialize from the successful contextual bidirectional checkpoint;
2. freeze GPT, the math tower, math-to-GPT and GPT receiver modules;
3. train only GPT-to-math and math receiver modules;
4. pair bridge-required complementary views with bridge-redundant shared views;
5. preserve bridge-disabled specialist logits when the frozen baseline is
   correct;
6. require correct messages to beat disabled and shuffled controls on required
   examples;
7. softly suppress redundant GPT-to-math gates without imposing global gate
   sparsity;
8. select checkpoints with greedy-generation causal panels and a hard
   shared-view no-harm constraint.

The full preregistration and implementation contract are recorded in
`V1_2_BRIDGE_REVISION.md`. Broad V2 math training follows only after this
bridge-control experiment establishes when information should and should not
cross.

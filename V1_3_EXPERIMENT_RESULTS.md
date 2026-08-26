# CFTN-Text V1.3 experiment results

Status: the original preregistered 12-stage pipeline completed stages 1-9 and
failed acceptance at Stage 10. It did not run the registered Stage-11 sealed
causal suite or Stage-12 evidence assembly. A subsequent, separately identified
recovery program passed native end-to-end acceptance by replacing generated
latent requests and learned answer composition with learned finite-intent
dispatch, lossless typed request compilation, and deterministic result
composition.

This distinction is permanent: the original pipeline is **failed/incomplete**,
while the recovered typed runtime is **passed within the registered V1.3 task
grammar**. The recovery result does not retroactively turn the missing original
stages into completed stages.

Preregistered design: `V1_3_EXPERIMENT_PLAN.md`

Machine-readable recovery ledger:
`evidence/v1_3_recovery_final_evidence.json`

Detailed hard-transition findings:
`docs/V1_3_STAGE10_HARDENING_FINDINGS.md`

## Executive conclusion

V1.3 established four useful results.

1. The math and exact-string specialists passed their task-matched native
   capability gates.
2. Stage 9 demonstrated strong supervised soft routing and causally useful
   multi-specialist communication.
3. Stage 10 falsified the assumption that thresholding the learned soft latent
   path would produce a reliable conditional-compute runtime. Three hardening
   attempts failed, and the registered stages after Stage 10 did not run.
4. The recovery program demonstrated a successful alternative: learn only a
   finite public-prompt intent, copy operands losslessly into typed specialist
   requests, execute only the requested towers, and compose exact payloads
   deterministically. This runtime passed all final native gates without oracle
   metadata.

The accepted V1.3 runtime therefore bypasses the legacy learned latent
wake/halt request path and does not rely on the learned answer composer. Those
components remain valuable failed experiments and compatibility artifacts, not
accepted inference mechanisms.

## Immutable provenance

- V1.3 experiment revision SHA-256:
  `2e67634056aba0401af71cb590713cd24b58d8ba28e48f6c0350d659140dc481`
- V1.3 manifest SHA-256:
  `7eeb8f8992910a3381f6009bf5c68c9936a3321aae59d28a02a8b2def0c61e80`
- V1.2 initialization checkpoint SHA-256:
  `6d3d13eefdcb17f6c848c11baf69f4f60040a8b16f0da34c4882fa2dd3235082`
- V1.2 final report SHA-256:
  `3dfaa77f318520bddcddb1a14bf486307991c626de05e7126366d3e7aa2bdec2`
- Math-specialist checkpoint SHA-256:
  `97a48a484157f10e9ce41a8d9fef8ea8b168180d26ebee00d50f549bc506eb27`
- String-specialist checkpoint SHA-256:
  `8a8e102f011b23ed8aa00d8816124f0d868e1ac90c8f4712870c38471b62a70c`
- Stage-9 soft checkpoint SHA-256:
  `71e1bf670097f494b722576f051908f27568446e4b4dc5b8b5e291558f4fb2c8`
- Adapter-continuation checkpoint SHA-256:
  `236d3b4b71b595cf99f6babf2d17f090823f5a2b716e32c671487a471becdb99`
- Fusion-recovery checkpoint SHA-256:
  `231566e1d17dca7a35bd7028c7af38bb9f450c38ff9f727fa7d278dcbb8cd790`
- Accepted collaboration/answer-bus checkpoint SHA-256:
  `cd5ef65a948607f53629bfd0e3837f3e6966c0497fbccfe15e11bb8f270f9ccf`
- Learned dispatcher checkpoint SHA-256:
  `82c166ecada2d5aa148bdd63b6d205ad976adc44493b2e159d40aea859b7f6d0`

The key checkpoint files were re-hashed against these values on 2026-08-27.
The original failed logs and checkpoints remain preserved under
`G:\ctfn-text\artifacts\v1_3_multi_specialist`.

## Registered pipeline ledger

| Stage | Registered action | Terminal result |
| ---: | --- | --- |
| 1 | Audit V1.2 pass | Passed |
| 2 | Prepare deterministic V1.3 data | Passed |
| 3 | Calibrate frozen GPT language path | Passed after registered interface repair |
| 4 | Train exact-string specialist | Completed |
| 5 | Seal native specialists | Passed after evaluator-budget repair |
| 6 | Train single-specialist capacity | Completed |
| 7 | Train dense mixed messages | Completed |
| 8 | Train dense recurrent communication | Completed |
| 9 | Train supervised soft wake | Completed; best checkpoint preserved |
| 10 | Harden wake decisions | Failed acceptance; no eligible checkpoint |
| 11 | Evaluate sealed causal suite | Not run |
| 12 | Assemble final registered evidence | Not run |

The raw `pipeline_status.json`, lock, and PID files still describe an old
Stage-10 process as running. They are preserved historical state, not evidence
of a live process. `sealed_evaluation/report.json` and
`v1_3_final_report.json` do not exist.

## Tower knowledge boundaries

The frozen GPT language calibration passed 1,000/1,000 examples under the
registered `first_nonempty_completion_line_v1` semantic protocol. This is a
calibration of the archival pure-language path, not evidence that GPT-2 is an
open-world instruction-following or factual chat model.

The math specialist passed every task-matched oracle-native primary cell at
100%. Its familiar V1.1 test accuracy was 99.98%, but its diagnostic held-out
language, compositional, and extrapolation splits were materially weaker. V1.3
therefore demonstrates the registered linear-equation interface, not broad
mathematics.

The exact-string specialist passed its familiar test at 99.76%. Task-matched
oracle-native coverage was 99.75% for exact string, 99.9% for multi-parallel,
and 98.8% for multi-sequential examples. Diagnostic extrapolation and held-out
paraphrase splits were much weaker and remain explicit capability boundaries.

## Stage 9: successful soft collaboration

The accepted Stage-9 checkpoint reported:

- GPT teacher-forced sequence accuracy: 84.22%
- GPT teacher-forced token accuracy: 86.115%
- exact required-set routing: 99.48%
- wake precision: 100%
- wake recall: 99.48%
- pure-language false wake: 0%
- causal message-loss gap: 8.1592

This was real evidence that the specialists and communication adapters could
help the coordinator under the trained soft execution regime. It was not proof
of discrete conditional execution because the soft phase executed every
specialist and layer normalization largely removed scalar wake attenuation.

## Stage 10: failed hard transition

Attempt 1 collapsed toward always-open routing. Attempt 2 initially retained
routing quality and then degraded. Attempt 3 isolated wake-gate BCE, physically
skipped sleeping specialists, froze the halt gate, and removed answer-utility
gradients from wake calibration. Even this corrected 10-epoch run failed:

- GPT sequence accuracy: 39.74%
- GPT token accuracy: 43.17%
- exact required-set routing: 60.00%
- wake precision: 100%
- wake recall: 45.38%
- no checkpoint eligible for promotion

Checkpoint-level diagnostics established that nominally closed soft messages
had still influenced layer-normalized receivers, inactive message slots were
marked valid, post-halt labels dominated the router objective, the third round
was unreachable, and the sequential curriculum accidentally contained only
one dependency order. These findings disproved the original hardening premise.

## Recovery sequence

### Oracle hard adapters and route falsification

Training adapters under oracle binary routing recovered protected explicit
math, language-dependent math, and multi-sequential performance. The selected
continuation checkpoint reached 66.7% exact-string accuracy but only 3.4%
multi-parallel sequence accuracy.

An exhaustive sweep evaluated all 84 one-, two-, and three-round schedules on
500 multi-parallel examples. No schedule materially repaired composition. The
best exact result was 3.4%, and it cost an extra specialist call while reducing
token accuracy. This ruled out route calibration or gate-only reinforcement
learning as the next fix because the available actions did not produce a
rewarding outcome.

### Fusion recovery

The next recovery changed the simultaneous-message consumer while preserving
the specialists and earlier checkpoints. It trained only message fusion and
GPT receivers, passed its optimizer/frozen-component contract, and selected an
acceptance-eligible epoch-4 checkpoint. This established a better fusion
initialization but did not by itself establish native end-to-end exact output.

### Clean answer-bus recovery and native mismatch

A typed lossless byte answer bus and pointer-copy answer composer were added.
On clean registered buses the selected checkpoint reached 100% generated
composer accuracy and passed its training acceptance gates.

The first native evaluation over 748 examples then failed:

| Task class | Native accuracy |
| --- | ---: |
| Exact string | 16.67% |
| Explicit math | 23.44% |
| Language-dependent math | 34.38% |
| Multi-parallel | 0% |
| Multi-sequential | 10.16% |

Answer-bus validity was 96.39%, showing that format validity did not imply task
transfer. A subsequent native/noisy-bus continuation also failed acceptance,
especially on reverse and multi-parallel composition. No continuation
checkpoint was promoted.

### Typed request and deterministic composition recovery

Direct tower diagnostics established that complete native specialist requests
could produce the expected payloads. The remaining failure was request
construction and learned composition, not missing tower capability on the
registered tasks.

The accepted recovery therefore:

1. learns only one finite intent and confidence from the public prompt;
2. copies quoted values, numeric operands, and dependencies from immutable
   prompt spans;
3. compiles typed calls into each specialist's independently validated native
   interface;
4. physically skips towers not present in the accepted call graph;
5. carries exact specialist payloads through typed result references; and
6. composes final results deterministically rather than decoding them through
   the failed learned answer composer.

## Learned dispatcher result

The final dispatcher used a confidence threshold of 0.9 and passed:

- 155,000 training examples;
- 5,500 validation examples at 100% accuracy and 100% coverage;
- 5,500 held-out paraphrases at 100% accuracy and 100% coverage; and
- 4,500 independent semantic/unsupported examples at 100% accuracy and 100%
  coverage.

Unsupported or structurally invalid prompts fail closed. The dispatcher never
generates operands and does not receive oracle task, wake, target, or
specialist-request metadata at runtime.

## Final native typed-dispatch result

The final public-prompt panel used the learned dispatcher, both native
specialists, lossless typed requests, no latent request messages, and
deterministic answer composition. It passed every acceptance gate:

- examples: 748
- oracle metadata visible to runtime: false
- dispatch-plan validity: 100%
- dispatch completion: 100%
- specialist payload validity: 100%
- answer-bus validity: 100%

| Task class | Examples | Accuracy |
| --- | ---: | ---: |
| Exact string | 300 | 99.33% |
| Explicit math | 64 | 100% |
| Language-dependent math | 64 | 100% |
| Multi-parallel | 192 | 99.48% |
| Multi-sequential | 128 | 100% |

The source native report SHA-256 is
`5020566c1fec7907e0ee617d5719881abf0b366b2efd7ce43ff252aae9d1071a`.
The deterministic typed-dispatch control report SHA-256 is
`bc8f5344281f5160848f0bcb0a85554c5e874dae504271945d53acd49bed7531`.
The learned-dispatcher training summary SHA-256 is
`e157fda5a09e1ca91fc5728ba48f52895c586121c193a1e1da99b12e73f2d40e`.

## Conditional-compute conclusion

The accepted typed runtime performs genuine conditional execution: only
towers named by the accepted finite call graph run, and disabling or omitting
an upstream tower makes dependent calls fail closed. This is stronger runtime
evidence than the rejected soft/thresholded wake path.

V1.3 did not complete the original preregistered compute-comparison Stage 11,
so it does not claim the missing dense-versus-conditional FLOP, latency, and
memory table. The accepted runtime records per-call execution and timings in
the inference trace, but that is a separate operational measurement.

## Final interpretation

The successful architectural lesson is not that latent communication is
always inferior. It is that exact external operands and exact specialist
results should not depend on a drifting latent interface when a lossless typed
channel is available. Semantic representations remain useful for choosing an
intent; copied typed spans and deterministic result references protect exact
execution.

V1.3 remains deliberately bounded. It demonstrates one-variable linear
equations, registered exact-string operations, their registered parallel and
sequential combinations, unsupported-request rejection, and the archival
pure-language calibration path. It does not demonstrate open-world dispatch,
broad mathematics, modern chat quality, or unregistered tool use.

## V2 requirements carried forward

V2 must preserve these safeguards:

- separate semantic intent prediction from lossless operand transport;
- make every public specialist request a finite typed call graph;
- fail closed on invalid, unsupported, or low-confidence requests;
- evaluate dispatch on registered, paraphrase, semantic, unsupported, and
  native end-to-end panels;
- keep exact result composition deterministic where the task contract permits;
- treat answer-bus validity and task accuracy as separate gates;
- separate wake calibration from halt calibration;
- require actual specialist skipping before claiming conditional compute; and
- never apply routing RL until the available discrete actions produce a
  measurable rewarding outcome.

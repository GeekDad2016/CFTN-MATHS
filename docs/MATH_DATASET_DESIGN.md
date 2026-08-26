# Math dataset design, supervision repair, and scaling evidence

Date: 2026-08-26. Status: implementation and bounded pilot; **not authorization
for a long training run, checkpoint promotion, or downstream pipeline release**.

## What the V1.3 / V2 audit established

V1.3 inherited a frozen V1.1 math tower trained on 100,000 integer linear
equations, with a consistent SUB/DIV worked trace and four numerical curriculum
bands. The V1.3 joint corpus trained interfaces/cooperation, not a new standalone
math solver. Its successful familiar-task result did not establish unrestricted
mathematical reasoning.

The archived standalone report for that inherited checkpoint records 99.98%
familiar-test generation accuracy, but 0.80% on held-out wording, 20.87% on larger
inputs, and 0.67% on out-of-range answers. Those are raw-question specialist
results, **not** a reevaluation of the final V1.3 dispatcher. Context was 256 byte
tokens; it is not established that length limits caused those generalization
failures.

The sealed V2 training corpus has 400,000 rows and 60 family labels:

| Source | Training rows | Actual supervision |
| --- | ---: | --- |
| CFTN generated | 150,000 | Six families; short, often abbreviated work |
| DeepMind Mathematics | 212,690 | Answer only |
| GSM8K | 7,473 | 7,378 calculator-annotation traces; 95 answer only |
| MathQA | 29,837 | Abstract operation program followed by answer text |

Thus 53.20% of stored rows are answer-only. These are **stored-row fractions**,
not the rebalanced recovery's update fractions. `full_trace_v1` uses the stored
target; it does not create missing work. The original V2 config has curriculum
phases, but the capacity-recovery override admitted all recorded levels from
the beginning.

The audit independently recomputed all 150,000 generated training answers and
5,000 generated validation answers with no mismatches. All training and broad
validation sequences fit the 4,096-byte context. The model really received the
complete public question; current standalone recovery did not substitute the
private/opaque joint-training view. No padding, causal-mask, or label-shift bug
was found in the bounded model probes.

Two-variable-system targets jumped directly to determinant, x and y, then
repeated x/y in the final payload. On a 48-case probe of the capacity epoch-5
checkpoint, teacher-forced token accuracy was 95.40%, but full-sequence accuracy
was 2.08%; determinant accuracy was 16.67%. Supplying correct work yielded a
correct final answer in 16/16 prefix interventions; supplying altered work made
the model copy the altered answer in 16/16. Copying is useful, but this shows why
accurate final copying is not evidence of correct preceding computation.

There is also genuine inconsistent supervision in some MathQA rows. Example:
64 frisbees at $3 or $4, $204 total, asking for the number at $3. The stored
program computes `64*3=192; 204-192=12`, then supplies answer `52`. It omitted
`64-12=52`. Record ID:
`9b53b20fef3e87c340f9f54c94320601023c727041f98ad3183b3b9e2d514e19`.

The earlier restricted audit found 5,178 training value-mismatch review flags,
including 567 programs that returned another listed option, also reproduced
from the actual linear target. These counts are **not a dataset-wide label error
rate**: rounding, units, unsupported semantics and bad questions need review.
Hash/schema integrity is not semantic correctness. Nor do these findings prove
that MathQA alone caused the other sources' performance shortfall.

Detailed original read-only evidence is preserved under
`C:\CFTN\.runpod\diagnostics\v1-v2-dataset-comparison-2026-08-26.md`,
`v1-v2-dataset-audit-2026-08-26.jsonl`, and
`mathqa-program-review-2026-08-26.jsonl`. The copied V1 data and RunPod audit
outputs remain in `/tmp/cftn-dataset-compare-20260826`; these temporary copies
are not the only evidence and should not be the durable archive.

## Implemented opt-in repair

The sealed `cftn_text/v2_data.py`, original corpus, acceptance thresholds and
production loss defaults are unchanged.

- `cftn_text/verified_math_data.py` adds `cftn_verified_procedure_v1` records.
  Inputs are parsed from **public question text only**. A new record links to
  its original record/content IDs and carries its own digest, explicit steps,
  final answer, and computation/copy spans.
- Two-variable systems show the determinant products/subtraction, both
  numerators, divisions, and recomputed equation residuals. A separate Gaussian
  elimination implementation checks the final result, rather than simply
  trusting the trace generator's Cramer's-rule result.
- Multiplication supports signed finite decimals. It scales to integer digits,
  computes distributive partial products, accumulates them, applies sign and
  rescales exactly. Direct rational multiplication independently verifies the
  final result. The initial bound is twelve digits per scaled operand. These
  are partial-product traces, **not yet full digit/carry microtraces**.
- Exact `Fraction` arithmetic checks all operations; zero division, oversized
  arithmetic, singular systems, unsupported grammar, wrong labels and corrupt
  traces fail closed. No answer-bearing metadata is exposed in the input.
- MathQA triage executes a restricted set of actual stored program operations.
  Unsupported/ambiguous bindings and program/answer mismatches are quarantined
  in a sidecar. Even an exact match is labelled
  `internally_consistent_needs_semantic_review`, **not** automatically certified
  as solving the question. No original label is automatically rewritten.
- `cftn_text/computation_supervision.py` provides an experimental per-example,
  per-role loss: 70% computed-result tokens, 20% copies, 10% formatting. It
  averages within each role first, renormalizes missing roles, and then averages
  examples. Longer traces cannot dominate solely by having more bytes. Prompt
  and padding remain masked. This is distinct from multiplying the entire
  answer suffix, including tags/EOS, by four.

Post-pilot code review found that a scalar's final fraction-to-decimal
representation conversion must be labelled **compute**, not copy. A regression
test now enforces this. The initial pilot corpus and results remain unchanged;
a newly built derivative has a different procedure-code/manifest hash. Do not
claim that the initial GPU pilot tested this later label correction. Also, the
current multiplication recipe still leaves decimal digit extraction/scaling
implicit before its partial products; explicit operand-binding/scaling lessons
are a next curriculum experiment, not a demonstrated fix.

The pilot excludes MathQA from **all** arms. It therefore tests procedural
supervision/loss, not the causal effect of removing MathQA. The broad original
acceptance tests have not been filtered or declared passed.

## Bounded comparison protocol

Run tests and model work on RunPod, not the workstation. Use a fresh clean Git
worktree. Preserve the source and every negative result.

Source: capacity-recovery epoch 5, global step 62,500:
`math_capacity_recovery_r3/checkpoint_epoch_0005.pth`, SHA-256
`2ddb776715b0ee0accfd03e2d98ea4f29cb47c7b4954c02a6beb759150357b08`.

Protected original `math/math.best.pth`:
`fe2c056a1ee1d4a3514537681d82124b0312f45c27f72f8e73d2afc747d53973`.

Parent manifest:
`a0a1cec180d5400faafe3e6794793b949dc08743ca9b2c7899a273a181ae21f0`.

`python -m tools.pilot_verified_math prepare` creates a fresh derivative directory;
it refuses an existing destination. It checks the parent manifest, unchanged
generator and the train/validation file hashes and record schemas. It records
which parent splits were audited rather than claiming a new full-corpus audit.
The derivative retains both original and worked variants of exactly the same
training questions. Numerically identical multiplication with reversed operands
cannot cross the train/validation boundary. Unsupported training rows and
duplicates are explicitly counted; validation parser failures abort, not filter.

`python -m tools.pilot_verified_math run` compares:

| Arm | Target | Objective |
| --- | --- | --- |
| Baseline | Zero updates from source | Evaluation only |
| Control | Original target | Existing 4x answer-suffix loss |
| Loss only | Original target | Per-example computation/copy/format loss |
| Verified | Verified worked target on two target families | Same new loss |

All training arms use the same source weights, fresh AdamW, seed, questions,
batch order, curriculum, learning rate and update count. The default is 300
steps per arm, batch 16; the tool refuses more than 600 steps. First third:
foundation magnitude band; remaining steps: all supported training magnitudes.
Every batch contains 25% replay from unchanged variables-both-sides and nested
parentheses tasks. This compares the target/loss interventions, not curriculum
against no curriculum. Equal updates/examples are **not equal target-token or
GPU-time budgets**; both are recorded.

Evaluation is greedy from the public question, with no gold work, answer
prefix, symbolic fallback or external execution at inference. Fixed panels:
64 systems, 64 multiplications, 32 cases per replay family, plus a 128-case
stratified unfiltered broad diagnostic. That small broad panel is not a
replacement for the original 1,024-case training gate or full evaluation.
All arms get the same 1,024-byte output budget. Teacher-forced computation and
first-result accuracy are separately labelled; the first operation changes
with trace representation and is not an apples-to-apples free-generation score.

Prespecified screening conditions for the verified arm:

1. At least +5 percentage points in mean targeted-family answer accuracy versus
   both controls and the zero-update baseline.
2. Neither targeted family regresses versus control/baseline.
3. At most 3 percentage points regression on each replay family and broad mini-panel.
4. No targeted generation hits the output cap.

A screen pass means a **larger confirmation is warranted**, not that the math
tower is fixed or meets 99%. Small panels and a single seed are exploratory.
No pilot checkpoint is named/promoted as `math.best.pth`; pilot files have an
explicit non-promotable format. There is no pipeline-launch code in this tool.
Pilot results, costs, generation rows and any negative findings belong in
`MATH_SUPERVISION_PILOT_RESULTS.md` after execution.

## Durable structure for larger math towers

Use the same principles before increasing width/depth or targeting a 4B tower:

1. **Define support.** Specify operations, numeric ranges, number representations,
   languages, units, answer forms and allowed approximation. A family name such
   as algebra or difficulty level 3 is not a sufficient curriculum definition.
2. **Separate problem, procedure and rendering.** Maintain a typed mathematical
   object/program, multiple public-question renderings, executable intermediate
   values, a final answer and verifier evidence. Do not use privileged answer
   metadata in the model prompt or evaluate with oracle dispatch.
3. **Verify independently.** Arithmetic checks alone do not prove the program
   answers the question. Verify operand binding, operations, constraints, units,
   final requested quantity, residuals and answer canonicalization. For imported
   data, quarantine unsupported, contradictory or ambiguous examples with IDs
   and reasons. Approximation requires an explicit versioned tolerance policy.
4. **Balance procedure and final-only tasks.** Provide worked supervision for
   learning algorithms, and eventually train/evaluate requested answer-only
   behavior too. Measure first erroneous computation, not just copied answer
   payloads or teacher-forced full-trace token accuracy.
5. **Progress by measured competence.** Single-digit operations/carry/borrow ->
   signed integers -> rational/decimal operations -> equations/systems ->
   polynomial manipulation -> functions/calculus -> probability/linear algebra
   -> advanced undergraduate mathematics -> graduate proofs/problem solving.
   Advance each supported band only after held-out free-generation gates, while
   retaining replay. The current pilot does not implement all these stages.
6. **Control mixture exposure.** Record actual sampled examples, source/family
   quotas, difficulty/magnitude/step counts and supervised-token totals, not
   merely stored dataset proportions. Ensure adequate per-skill repetition.
7. **Split before rendering.** Partition by canonical mathematical object and
   derivation, not just text hash. Reserve independent wording, larger operands,
   new answers, operation compositions and problem structures. Equivalent
   equations, swapped operands and paraphrases must not leak across splits.
8. **Keep evaluation immutable.** Training cleanup must not silently remove hard
   evaluation cases. Report unsupported/verifier-failed/capped rows in the
   denominator. Add separately versioned corrected benchmarks when needed,
   reporting original and revised results side by side.
9. **Budget context explicitly.** Count complete UTF-8 prefix + trace + EOS;
   never silently truncate. Count output-budget hits separately. Long worked
   solutions require larger generation budgets and compute accounting.
10. **Track transfer and cost.** Measure native greedy accuracy, canonical and
    mathematically equivalent answers, per-family validity, procedure correctness,
    replay regressions, first-computation errors, runtime, generated tokens,
    peak memory and failed-case IDs. Eventually compare base Qwen, standalone
    math, routed CFTN, and disabled-math ablations on the same saved questions.
11. **Seal and reproduce.** New corpus version, parent hashes, source revisions,
    licenses, verifier/code hashes, exclusion ledger, split hashes, curriculum,
    loss definition, model/tokenizer config and checkpoint identity. Never
    re-sign a different corpus as the old one. Keep generation provenance
    separate from mutable runtime paths/curriculum settings.
12. **Scale only with evidence.** Short fixed-budget ablations first, then
    multi-seed/longer confirmation with unchanged acceptance. For proof-level
    mathematics, add proof-aware/formal or expert-reviewed verification rather
    than treating plausible text as a verified solution. Distillation teachers
    may propose work; their output is not ground truth without checking.

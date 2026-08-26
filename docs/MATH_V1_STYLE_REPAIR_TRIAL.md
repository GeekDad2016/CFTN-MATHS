# V1-style V2 math repair: bounded three-epoch trial

## Decision and evidence, 2026-08-26

Stop adding comparison arms. Implement one coherent training recipe, measure
three epochs, and preserve the result whether it succeeds or fails.

V1.3 inherited the V1.1 frozen math specialist. Its familiar integer linear
equations reached 99.98% generated-answer accuracy, with one consistent SUB/DIV
worked solution and explicit numerical bands. Raw-question generalization was
much weaker (0.80% held-out wording, 20.87% input extrapolation); these are not
the later dispatcher results. V1 demonstrated mastery of its trained support,
not broad mathematics. Its 256-token context is not proof of the OOD cause.

V2's input, causal labels, padding and gradients were checked: the public question
reaches the tower, all 24 layers receive gradients, and sampled full examples fit
4096 tokens. The capacity recovery admitted all recorded difficulty levels from
epoch one. Supervision mixed short worked targets, answer-only imports and some
contradictory MathQA program/answer pairs. Correct input plumbing alone did not
make that learning signal sufficient.

The first long-work pilot failed: frequent wrong operand binding and incomplete
generated traces, without native transfer. The short prerequisite pilot proved
tiny-set recall (28/28 in each arm), but held-out arithmetic remained weak:
compact-worked multiplication/subtraction 65.625%, division 93.75%. Neither arm
passed all prerequisite gates. These results justify a coherent graded math
recipe; they do not establish that it will repair V2.

## Implemented change

- New versioned corpus in `cftn_text/v2_school_data.py`; original V2 generator,
  datasets and sealed manifests remain untouched.
- Real signed addition, subtraction, multiplication, exact division and linear
  equation questions. No operand-extraction or syntax-only training tasks.
- Linear equations use V1's exact canonical SUB/DIV trace. Multiplication adds
  decimal-place decomposition before the final sum. Every result is recomputed
  from the public question using exact arithmetic; wrong or hidden overrides fail.
- Three numeric bands. Linear bands reproduce V1's first three coefficient,
  solution and offset limits: (8,20,50), (16,50,125), (32,100,250). Other families
  start with operands up to 15, then 99 and 999; exact division bounds its divisor
  and quotient. These are declared supported ranges, not a claim about all math.
- Mathematical objects split before wording; commutative swaps and scalar
  multiples of a linear equation share a split. Three wording styles train;
  a fourth is held out. This is not exhaustive symbolic equivalence grouping.
- Each epoch samples 16,384 examples: 75% balanced new school problems and 25%
  unchanged variables-both-sides/nested-parentheses replay. The elementary finite
  support is repeated deliberately. Larger bands retain 30% earlier-band replay.
- Per-example role loss: 50% new computed values, 25% copied values, 25% format.
  No hidden answers enter model input. Fresh optimizer/scheduler, unchanged
  24-layer byte tower and tokenizer, checkpoint-5 source, LR capped at 1e-4.
- Advance only after at least two epochs in a band AND every current family has
  >=99% generated answers, 100% intact/EOS-terminated format, >=95% exact work,
  zero generation caps, and replay loss <=3 percentage points. Failed gates
  retain the current band. Production acceptance is never changed or implied.

The three-epoch budget is a directional training trial, not a replacement for
V1's much longer mastery training. Rational/decimal systems, imported advanced
math and PhD-level material remain excluded until their prerequisites pass.

## Evaluation and preservation

`tools/train_v2_verified_school.py` trains exactly one model for three epochs.
Each epoch evaluates 64 held-out objects per current family, plus wording,
next-band and unchanged replay diagnostics. Baseline/final legacy multiplication,
systems and broad panels are paired at a 512-token generation budget; this short
diagnostic is explicitly not the original production acceptance evaluation.
School decoding is capped at 256 new tokens, including recorded EOS/cap evidence.

Save per-example generations, corpus and verifier hashes, sampled family counts,
training averages, generated-answer/exact-work metrics, timing, tokens, peak
memory and a hashed checkpoint per epoch. Canonical JSON evidence uses atomic
replacement and readback verification after the earlier padded-JSONL incident.
No checkpoint is promoted and no remaining pipeline is launched by this runner.

Protected original SHA-256:
`fe2c056a1ee1d4a3514537681d82124b0312f45c27f72f8e73d2afc747d53973`.
Source capacity epoch-5 SHA-256:
`2ddb776715b0ee0accfd03e2d98ea4f29cb47c7b4954c02a6beb759150357b08`.
All acceptance tests and model execution run on RunPod, not the workstation.

Results will be appended after the bounded run. See
[dataset design](MATH_DATASET_DESIGN.md) for the future scaling structure and
[earlier pilot findings](MATH_SUPERVISION_PILOT_RESULTS.md) for preserved failures.

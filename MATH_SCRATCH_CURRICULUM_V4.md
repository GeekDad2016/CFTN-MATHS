# V2 math scratch curriculum V4

V4 is a new 24-layer byte-level math tower trained from random initialization.
It does not inherit a rejected checkpoint, use a frozen teacher, or skip phases.
The protected historical checkpoint remains read-only evidence and is not an
initialization source.

## Why this run exists

The earlier broad runs mixed easy verified procedures, advanced DeepMind
families, and answer-supervised language rows before a new tower had learned a
stable procedural representation. Token accuracy improved, but exact generated
traces stayed weak in many families. V1.3 and the bounded V2 repair pilots show
that the byte-level tower can learn when the question, procedure, and final
answer provide an exact shared training signal.

V4 tests that causal hypothesis directly:

1. Integer arithmetic and one-variable equations use only independently
   verified school rows and exact generated procedures.
2. Signed fractions and nested equations are added only after foundations pass.
3. Systems and mathematical word problems follow, with cumulative replay of
   every earlier verified skill.
4. DeepMind numeric families enter only after verified generated mathematics is
   stable.
5. DeepMind symbolic families enter after numeric algorithms.
6. The final phase consolidates every bucket and must pass the unchanged broad
   generation gates twice consecutively.

Low scores for future panels are diagnostic, not current-phase failures. Phase
advancement is based on generated answers, intact output format, and exact
procedural traces. No checkpoint becomes best/accepted before the final phase.

## Reproducibility and safety

- Training data is rebuilt into a new sealed derivative root and fully audited
  before CUDA allocation.
- The V4 contract pins the adopted parent manifest, its resolved configuration,
  and its generator source hash; a changed external corpus is a new dataset,
  never a silent substitute for the historical recovery corpus.
- MathQA program rows remain quarantined.
- The first two phases reject any imported DeepMind or GSM8K row.
- Every phase samples exactly 100,000 examples from disjoint, family-balanced
  quota groups.
- A phase advances only after two consecutive passes and fails closed at its
  maximum epoch budget.
- The historical protected checkpoint hash is verified before and after the run.
- All acceptance tests and CUDA smoke checks run on RunPod.

## One-command start or exact resume

From the persistent checkout on RunPod:

```bash
./scripts/start_or_resume_v2_math_scratch.sh
```

The wrapper invokes the V4 `auto` entry point. A fresh artifact starts from
random weights; an existing compatible artifact resumes only from its latest
checkpoint under the identical data manifest and resolved settings contract.
The only code-revision exception is the immutable validation-scope record
described below; it is limited to its named tested revision.

## Phase-scoped generation validation

Teacher-forced validation always covers the fixed mixed holdout. Autoregressive
generation is much slower, so a resumed artifact may carry an immutable
`generation_panel_scope_override.json`. It runs only the active phase's primary
panel plus every panel named by that phase's acceptance thresholds. This does
not remove a gate: the omitted panels are not acceptance criteria for that
phase, return when their own phase becomes active, and remain part of final
evaluation. The override records its source checkpoint, data manifest, and the
one compatible code revision; changing any of those values fails closed.

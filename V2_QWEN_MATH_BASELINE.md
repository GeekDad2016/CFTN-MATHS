# V2 frozen-Qwen math baseline

Date: 2026-08-24

This diagnostic measures how much of the V2 broad-math validation curriculum
the frozen coordinator can solve directly when it receives the raw numeric
problem. It does not evaluate dispatcher routing or the sealed CFTN interface,
and it does not replace standalone math-tower acceptance.

## Immutable setup

- Model: dense `Qwen/Qwen3-4B-Instruct-2507`
- Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Parameters: 4,022,468,096, all frozen
- Precision: BF16
- Decoding: greedy, no external tools
- Panel: 36 validation records, exactly 12 from each configured difficulty,
  selected deterministically by source/family round robin
- Panel record hash:
  `d2a879d368ad20d54349873186c5f0fa65a75b61875cc631d034c27fdece6666`
- Scoring: the same `<answer>...</answer>` extraction and symbolic-equivalence
  scorer used by V2 generation evaluation

## Results

| Input/prompt condition | Token cap | Correct | Valid answer tags | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Sealed coordinator view with numeric slots hidden | 256 | 2/36 (5.6%) | 17/36 | Interface control only; Qwen cannot reconstruct hidden operands. |
| Raw problem, answer only | 256 | 12/36 (33.3%) | 30/36 | Suppressing explicit reasoning substantially harms this model. |
| Raw problem, concise reasoning | 512 | 32/36 (88.9%) | 32/36 | All four misses reached the generation cap before an answer tag. |
| Raw problem, concise reasoning | 2,048 | 33/36 (91.7%) | 33/36 | Operational baseline used for curriculum comparison. |

In the 2,048-token condition, Qwen scored 12/12 on difficulty 1, 9/12 on
difficulty 2, and 12/12 on difficulty 3. It solved all 26 sampled
project-generated records and 7/10 sampled DeepMind Mathematics records. The
three unresolved records were two polynomial-factorization tasks and one
cubic-sequence nth-term task. They again exhausted the generation budget
without a final answer tag, so 91.7% is a bounded operational score rather than
proof that all three answers were mathematically impossible for Qwen.

The answer-only control must not be used as the base-capability estimate. It
shows that the coordinator needs a reasoning allowance when direct solving is
measured. Conversely, the 314-second runtime of the 2,048-token panel shows why
correct conditional routing can still be valuable even when Qwen is capable:
a specialist may provide a shorter, cheaper, more reliably typed execution
path.

## Curriculum decision

Do not mutate the active, hash-sealed V2 run. Its scratch-trained 19M-parameter
byte tower still needs the foundational phases to learn its own representation;
Qwen's pretrained competence does not transfer into that tower's weights.

For the next sealed data/config revision:

1. Keep a shorter but nonzero foundation phase as a prerequisite and regression
   set.
2. Increase late-phase weight on exact high-degree polynomial factorization,
   rational coefficients, repeated roots, and independently verified symbolic
   equivalence.
3. Add composed calculus instructions that require tracking named intermediate
   functions and derivative order before simplifying or factoring.
4. Add finite-difference sequence induction, recurrence discovery, and nth-term
   extrapolation with adversarial near-patterns.
5. Expand exact arithmetic in non-decimal bases, fractions, units, number
   theory, probability, and multi-step mixed-family problems.
6. Build a larger frozen-Qwen comparison panel of at least 100 examples per
   difficulty, separated by source and family, before locking the revision.
7. Select specialist-routing targets from tasks where the tower is more
   accurate, materially cheaper, or more format-reliable than Qwen—not merely
   from tasks labelled `math`.
8. Require standalone generated-answer accuracy for both Qwen and the tower on
   the same sealed hard panel. Teacher-forced tower metrics are not directly
   comparable with this free-generation Qwen baseline.

The diagnostic evaluator is `tools/evaluate_v2_qwen_baseline.py`. Example:

```bash
python -m tools.evaluate_v2_qwen_baseline \
  --config config/v2_broad_math.yaml \
  --split validation \
  --examples-per-difficulty 12 \
  --batch-size 4 \
  --max-new-tokens 2048 \
  --prompt-mode brief_reasoning \
  --device cuda \
  --output-root /workspace/cftn-text/diagnostics/qwen_math_baseline
```

Use `--prompt-mode answer_only` only as a matched control. The evaluator reads
the raw `problem` field deliberately; using `gpt_problem` instead measures the
sealed coordinator interface, not Qwen's standalone mathematical knowledge.

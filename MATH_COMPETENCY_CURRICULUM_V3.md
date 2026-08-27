# V2 math competency curriculum v3

Date: 2026-08-27. This is a fresh fail-closed experiment. It does not rewrite
the terminal v2 run, promote a production checkpoint, or release downstream
V2 stages.

## Why v2 stopped

`math_competency_curriculum_v2` correctly stopped after phase 1 epoch 8. All
16 current-phase generation/format/trace gates passed, including 100% school
answers and 92.19% held-out wording, but broad generated-answer retention fell
from the protected 29.30% source baseline to 11.33%. The required floor was
26.30%. The result is therefore `failed_acceptance`, not a usable checkpoint.

This distinguishes two responsibilities:

- The dispatcher and gates learn whether a prompt needs mathematics, which
  tower(s) to call, dependencies, and when to return to the language model.
- The math tower still needs enough mathematical language to bind the public
  question to operands, operations, constraints and requested output. Moving
  all language understanding into the dispatcher would turn it into an oracle
  solver and would not teach the tower to solve the request it receives.

The v2 failure was not that the first skill was unlearnable. It was already
mastered and was trained too aggressively again: 400,000 examples per epoch at
up to 1e-4 learning rate. The local task improved while broad abilities were
erased.

## Controlled v3 change

`config/v2_full_supervision_v3.json` keeps the same 24-layer tower, sealed data,
public-question input, full-trace objective and acceptance thresholds. It
changes only progression and preservation:

1. Before any optimizer update, each non-final phase is evaluated with its
   complete unchanged generation and retention gates. A phase is skipped only
   when every gate passes. The first unmet phase becomes the training phase.
2. Epoch exposure falls from 400,000 to 100,000 sampled examples and peak
   learning rate falls from 1e-4 to 2e-5.
3. Every phase is cumulative. Previously introduced generated procedures,
   school problems, DeepMind numeric/symbolic families and GSM8K language
   examples remain in explicit disjoint replay buckets.
4. A frozen copy of the source checkpoint supplies a weight-0.1 KL preservation
   loss on DeepMind/GSM8K rows, but only where that source model predicted the
   complete supervised target correctly. Incorrect teacher behaviour is never
   distilled as truth.
5. Phase advancement still requires two consecutive complete free-generation
   passes. The final phase cannot be skipped and cannot promote a checkpoint
   until all final gates pass.

The ordered curriculum is:

| Phase | Maximum phase epochs | New emphasis |
| --- | ---: | --- |
| Verified integer foundations | 3 | Basic operation binding and one-variable equations |
| Verified rational and nested | 4 | Signed fractions and nested procedures |
| Verified systems and language | 5 | Systems and multi-step wording |
| Broad numeric algorithms | 8 | DeepMind arithmetic, comparison, measurement and number algorithms |
| Broad symbolic algorithms | 10 | Algebra, polynomial, calculus and probability transformations |
| Broad consolidation | 12 | Joint replay and unchanged final acceptance |

The maximum is 42 epochs, but zero-update entrance checks and competency gates
may make the run shorter. A maximum is not a promise to train every phase for
its full allowance.

## One-command start or resume

After the tested revision is present in the persistent RunPod checkout and no
CFTN trainer is active:

```bash
cd /workspace/CFTN-MATHS
/opt/cftn-data-pilot-venv/bin/python -m tools.run_v2_math_curriculum
```

The command now defaults to v3. On a fresh v3 artifact it starts from the
newest preserved `math_full_supervision_v1` checkpoint with fresh optimizer,
scheduler and curriculum state. On an interrupted compatible v3 artifact it
resumes the latest checkpoint. It refuses dirty code, duplicate trainers,
changed source/settings/data hashes, terminal failure, or incompatible model
architecture.

## Evidence to retain for future larger towers

For every later curriculum, preserve the source checkpoint and zero-update
panel, then record sampled updates by source/family, supervised role, operand
range, wording form and procedure length. Advance from concrete arithmetic to
rationals, equations, systems, symbolic manipulation and advanced mathematics
only on held-out free-generation competence with cumulative replay. A falling
training loss or teacher-forced token score is diagnostic; it is not mastery.
The decisive evidence remains generated answers, first erroneous computation,
exact trace, format validity and protected retention on previously learned
skills.

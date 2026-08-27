# V2 competency-gated math curriculum

> Archived result: this curriculum terminated `failed_acceptance` after its
> first phase because broad generated-answer retention fell below the protected
> floor. The successor and current default command are documented in
> [MATH_COMPETENCY_CURRICULUM_V3.md](MATH_COMPETENCY_CURRICULUM_V3.md).

This is the versioned successor to `math_full_supervision_v1`. It does not
rewrite that run, its data, or its checkpoints. The previous run remains valid
evidence for its original fixed curriculum.

## Evidence behind the change

- V1.3 reached 99.98% only on its narrow, canonical equation support. Its
  held-out wording, larger-input and out-of-range results prove that this was
  not broad mathematical generalization.
- The 24-layer V2 tower receives the intended public question and complete
  target. Padding, causal labels, context length and gradients were checked.
- In the first full V2 run, school generation reached 100% while broad
  generation remained around 23-25% as training loss continued to fall.
- Correct supplied intermediate work caused correct answers, while altered work
  caused the altered answer to be copied. This is evidence for local
  trace-following without reliable computation.
- Therefore more rows or more depth alone are not the next controlled change.
  The next experiment must change exposure and progression while preserving the
  same tower and sealed corpus.

## What V2 changes

`config/v2_full_supervision_v2.json` keeps the 24x384, 47.4M-parameter byte
tower and the sealed 504,430-row derivative corpus. It changes runtime sampling
and progression:

1. Each epoch is assembled from disjoint, named skill buckets rather than one
   source-level mixture. The trainer fails if bucket filters overlap.
2. Verified school and generated procedural examples make up at least 70% of
   the first three phases and at least 50% thereafter. Quarantined MathQA rows
   are forbidden.
3. Curriculum order is procedural: integer foundations; rational/nested work;
   systems/language; DeepMind numeric cohorts; DeepMind symbolic cohorts; full
   consolidation.
4. A phase advances only after at least two consecutive passing free-generation
   validations and its minimum phase-epoch count. It fails closed at its phase
   maximum. A fixed epoch number never grants advancement.
5. Generation is reported separately for school, unseen wording, the three
   verified generated cohorts, DeepMind numeric, DeepMind symbolic, and the
   unchanged broad holdout.
6. Best-checkpoint promotion remains disabled before the final phase. Final
   acceptance still requires the unchanged broad, source, difficulty,
   specialist-family, school and wording gates.

The DeepMind rows remain published-answer supervision, not verified worked
solutions. This version deliberately stages and limits them; it does not label
fabricated reasoning as truth. A later corpus version should add independently
verified procedures or executable intermediate checks family by family. Until
then, success on those panels is empirical answer-generation evidence, not
proof that every intermediate algorithm was learned.

## One-command start or resume

To target this archived v2 artifact explicitly (normally for forensic resume
only), use:

```bash
cd /workspace/CFTN-MATHS
/opt/cftn-data-pilot-venv/bin/python -m tools.run_v2_math_curriculum \
  --output /workspace/cftn-text/artifacts/v2_broad_math_400k_r4/math_competency_curriculum_v2 \
  --work /tmp/cftn-math-competency-v2 \
  --settings config/v2_full_supervision_v2.json
```

The same command has exactly two safe behaviours:

- If `math_competency_curriculum_v2` does not exist, it starts a new run from
  the newest preserved `math_full_supervision_v1/checkpoint_epoch_*.pth`, using
  only its model weights and a fresh optimizer, scheduler and curriculum state.
- If the V2 artifact exists with the exact same tested revision, settings,
  manifest and source hashes, it resumes its latest checkpoint exactly.

It refuses duplicate trainers, a dirty checkout, changed hashes, incompatible
architecture, incomplete artifacts, terminal failed acceptance, or a changed
curriculum. Logs, launcher records, checkpoints, W&B metadata and failures are
written to the persistent artifact root. The protected original checkpoint is
verified before and after training and is never overwritten.

The current `math_full_supervision_v1` process must be allowed to finish or be
safely stopped before running this command. Updating Git does not mutate the
already-running Python process.

# Full math supervision repair (26 August 2026)

The three-epoch school trial established learnability, not broad mastery:
generated current-band answers improved from 13.13% to 92.81%. Multiplication
was 81.25%; held-out wording and broad native retention still failed. Its
roughly 3,973 unique foundational objects cannot establish open-world
generalization. The role-weighted objective is provisional, not an isolated
ablation win. Do not present the small trial as production acceptance.

## Full run

- Same 24-layer, 47.4M-parameter byte tower; no architecture/tokenizer change.
- Warm-start the immutable school epoch-3 candidate (SHA-256
  `ebe0c1cd120234caf45d445ff5ac4f04c6a2ebecd8d79828bb7838657eb64e53`).
  Fresh optimizer/scheduler; this is a warm start, not exact resumption.
- Build a new derivative of the complete sealed 400,000-row training set.
  Never re-sign the original manifest. Keep every original evaluation file
  byte-identical, including MathQA. Test sets are not used for epoch selection.
- Expand all six generated families into short computed steps, binding only
  the public prompt and checking exact arithmetic/equation residuals. Systems
  explicitly compute determinant/numerators before x/y; nested and rational
  equations expose missing inverse steps; word problems expose each event.
- Retain published DeepMind/GSM8K supervision, including available GSM8K work;
  do not invent intermediate solutions for unsupported tasks. Such answers
  remain source-supervised, not independently certified. Keep provenance.
- Quarantine the entire MathQA program-training pool pending reliable semantic
  verification, including internally consistent programs whose number binding
  remains unverified. Preserve original records plus per-record reasons. This
  is NOT evidence that every quarantined answer is wrong. Hard evaluation stays.
- Add up to 12,000 distinct mathematical objects per family/numerical band,
  within finite support. Balance five school families and use seven training
  wordings. Hold out mathematical objects before wording; keep old held-out
  wording 3 and new wording 8 out of training. Exclude recognized parent math
  objects from the synthetic pool and check exact-question overlap globally.
  This is a bounded overlap audit, not a proof of semantic deduplication for
  every unsupported imported grammar.
- Loss: 50% computed/source-supervised result, 25% copied values, 25% syntax,
  averaged per role per example. Unsupported published work is not falsely
  annotated as independently verified intermediate arithmetic.
- 400,000 sampled examples per epoch, maximum 100 epochs. This counts sampling,
  not 400,000 unique new problems. Family-balanced source quotas retain broad
  replay throughout; school numerical ceilings advance at epochs 21 and 41
  only after passing the preceding gates. Broad consolidation starts at 61.
- Deadlines 20/40/60 require >=99% generated school answers per family, 100%
  validity with clean EOS, >=95% exact worked traces, and <=3 percentage-point
  broad regression from the new run's zero-update source. Failure stops closed.
- Final candidate gates retain the capacity run's broad source/family floors,
  plus school retention and held-out wording. Checkpoint selection is disabled
  before the final phase; all durable epoch checkpoints are retained. Only two
  scratch checkpoints are kept so the disposable 30GB filesystem cannot fill.
- Epoch validation includes unchanged 12,000-row teacher-forced validation,
  fixed school/wording panels and 512 broad greedy generations (512-token cap).
  Those compact native panels are curriculum screens, NOT substitutes for the
  unchanged full 2,048-token native production evaluation. No downstream
  pipeline is released automatically by this training run.

## Reproducibility and operations

`tools.train_v2_full_supervision prepare` audits the sealed parent, constructs a
fresh dataset directory, and audits the derivative. `launch` requires a clean
tested checkout, exact protected/source hashes, idle GPU, no other trainer,
fresh targets, and a process lock. `run --resume` requires identical config,
manifest and contract; ordinary warm starts explicitly reset optimizer state.
Local edits are committed and pushed; all tests and training execute on RunPod.
The SSH dashboard recognizes `math_full_supervision*`, exposes the full
curriculum/validation trends, and refreshes every 30 seconds. No new heartbeat
or parallel experiment is needed.

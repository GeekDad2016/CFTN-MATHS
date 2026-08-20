# V1.3 Stage 10 hard-wake findings

Status: recovery attempt 3 completed 10 epochs on 2026-08-18 and failed
acceptance. A replacement binary-routing recovery was designed on 2026-08-20
after checkpoint-level oracle and message-leakage diagnostics disproved the
earlier narrow gate-calibration hypothesis.

## Preserved evidence

- Stage 9 soft checkpoint:
  `supervised_soft_wake/supervised_soft_wake.best.pth`
  (`71e1bf670097f494b722576f051908f27568446e4b4dc5b8b5e291558f4fb2c8`).
- Stage 9 soft validation: 84.22% GPT sequence accuracy, 86.115% token
  accuracy, 99.48% exact required-set accuracy, 100% wake precision, 99.48%
  wake recall, and 0% pure-language false wake.
- Zero-update hard threshold baseline: 59.10% sequence accuracy, 88.84%
  exact required-set accuracy, 85.56% precision, 99.46% recall, and 0%
  pure-language false wake.
- Failed attempt 1 is preserved as
  `hardened_wake_attempt1_always_open_collapse`. Its epoch-3 checkpoint reached
  77.86% sequence accuracy while routing collapsed to 18.94% exact set, 49.54%
  precision, and 100% pure-language false wake.
- Failed attempt 2 is preserved as
  `hardened_wake_attempt2_task_gradient_drift_stopped_epoch3`. Epoch 1 retained
  88.74% exact routing, but epoch 2 fell to 56.40% exact routing and 71.40%
  pure-language false wake. It was stopped during epoch 3 before another
  checkpoint was written.

No checkpoint from either failed attempt is eligible for Stage 11.

## What was wrong

1. Stage 10 made only gate parameters trainable, but the straight-through wake
   activations still received gradients from GPT answer loss, specialist loss,
   causal-utility loss, preservation loss, and compute loss. Correct answers
   often became easier when extra specialists opened, so answer utility fought
   the supervised required-set target.
2. The original hard path did not skip sleeping specialists during training.
   Enabling true conditional execution exposed a BF16/FP32 scatter mismatch in
   mixed active/asleep batches. The previous run therefore had not tested the
   runtime mechanism it intended to deploy.
3. Wake hardening and halt hardening were coupled. On a fixed 128-example
   Stage-9 panel, applying the uncalibrated hard halt reduced required-wake
   recall to 90.48% and GPT sequence accuracy to 42.97%. On the same panel with
   hard halt disabled, recall recovered to 100% and sequence accuracy to
   54.69%. Specialist skipping itself did not change token or sequence accuracy;
   the regression came from early halting.
4. Checkpoint selection originally rewarded answer accuracy and causal message
   dependence without making routing thresholds mandatory. Attempt 1 could
   therefore select a checkpoint with 100% pure-language false wake.

## Minimal recovery change

Recovery attempt 3 changes only the hard-transition control policy:

- freeze GPT, specialists, bridges, receivers, and the halt gate;
- train only `wake_gates` at the inherited Stage-9 tail LR (peak 5e-7, no LR
  restart);
- optimize only supervised wake required-set BCE in Stage 10;
- disable answer/specialist/causal/preservation/compute gradients through wake
  decisions during this calibration phase;
- execute only specialists whose hard wake is active;
- keep hard halting disabled until a separate zero-update and calibration test;
- fail checkpoint promotion unless exact routing >=90%, precision >=90%, recall
  >=95%, and pure-language false wake <=5%; and
- report pre-halt routing separately when hard halt is explicitly diagnosed.

The corrected actual-checkpoint BF16 backward probe produced finite gradients
for exactly six `wake_gates` tensors (198,914 trainable parameters). Its total
loss equaled wake BCE (0.19536); answer, specialist, halt, causal, preservation,
and compute terms were diagnostic only and contributed no gradient.

The implementation regression suite passed 17/17 tests. At live step 100,
Stage 10 reported one optimizer group at LR 4.99998e-7, routing calibration
enabled on 100% of steps, auxiliary work on 0%, and exact equality between
total loss, model loss, and wake loss (0.13760 rolling average). GPU memory was
about 4.1/12.3 GiB because sleeping specialists are now actually skipped.

The 1,024-example zero-update panel with true specialist skipping and hard halt
disabled scored 57.81% sequence accuracy, 90.63% exact routing, 87.69%
precision, 99.71% recall, and 0% pure-language false wake. The subsequent
10-epoch gate-only run disproved the conclusion that the remaining job was
narrow: final exact routing fell to 60.00%, recall to 45.38%, and sequence
accuracy to 39.74%. No checkpoint was eligible.

## Root cause established after attempt 3

1. The soft phase multiplied wake probabilities into request and return
   messages before `GatedCrossReceiver.message_norm`. Layer normalization
   largely removes scalar attenuation, so every nonzero sigmoid activation can
   carry a substantial message. The soft phase also executes every specialist.
   Its thresholded routing metrics therefore did not represent conditional
   communication or compute.
2. Inactive return slots were concatenated as zero tensors but marked valid by
   an all-ones message mask. An actual Stage-9 checkpoint showed a maximum
   84-logit difference between this all-closed path and the receiver-bypassed
   path. Hard-closed communication was not a no-op.
3. On a balanced 256-example diagnostic, Stage-9 soft routing scored 85.16%
   sequence accuracy. Keeping nominally closed leakage while forcing required
   activations to one scored 85.55%; removing the leakage scored 53.52%.
   Perfect oracle binary routing therefore could not rescue the adapters.
4. The unmasked wake BCE contained 380,000 post-halt negatives in the
   100,000-example training set. Those unreachable labels were 76% of all
   negatives and drove the gate-only recovery toward closed gates.
5. No example requires round three. A two-round checkpoint panel improved
   routing over three rounds without training.
6. The intended alternating sequential curriculum was broken: every
   `multi_sequential` index is odd, so `index % 2` generated 10,000
   string-then-math training examples and zero math-then-string examples.

## Binary-routing recovery contract

- Preserve the sealed Stage-9 checkpoint as initialization and reset optimizer
  state.
- Use two reachable rounds and keep hard halt disabled.
- Mask inactive message tokens and bypass all-masked receiver rows exactly.
- First train only bridges and receivers under oracle 0/1 routing with physical
  specialist skipping.
- Then freeze adapters and train only wake gates plus a zero-initialized round
  embedding on reachable-round supervised labels.
- Never send task, causal, preservation, halt, or compute gradients into the
  router-calibration phase.
- Derive balanced sequential orders deterministically from the sealed manifest
  and record the derived record-ID hashes without altering the source files.
- Stop on validation patience and promote routing checkpoints only after exact
  set >=95%, precision >=95%, recall >=98%, false wake <=1%, and collapse
  guards pass.

## Required V2 safeguards

Before V2 reaches its own hard transition:

1. Separate wake-gate calibration from halt-gate calibration in the stage
   contract and optimizer groups.
2. During wake calibration, detach or zero all answer-utility gradients into
   discrete wake decisions; use supervised routing targets first.
3. Test true conditional execution before training, including mixed active and
   sleeping rows under BF16.
4. Require sleeping paths to be behaviorally harmless and test zero-message
   versus receiver-disabled output.
5. Evaluate a zero-update hard baseline for each discrete control independently:
   wake threshold, specialist skipping, and halt threshold.
6. Make checkpoint eligibility fail closed on exact routing, precision, recall,
   false wake, no-harm, and causal dependence before ranking eligible models by
   answer quality.
7. Report routing by specialist, task class, and round, and distinguish raw
   pre-halt predictions from runtime-reachable predictions.
8. Do not claim conditional-compute success from message masking alone; record
   actual specialist executions and compute saved.

## Oracle-adapter recovery outcome and continuation (2026-08-20)

The first oracle-binary adapter recovery stopped normally after epoch 4 when
the legacy selector reached patience 3. Training was real: all 180 trainable
adapter tensors changed between epochs 1 and 2 (7,029,208 changed elements),
and optimizer state existed for every trainable tensor. Validation token
accuracy improved from 69.995% at epoch 1 to 71.413% at epoch 4 and validation
loss fell from 3.83025 to 3.55680. Strict sequence accuracy stayed near 63.6%,
while causal message-loss gap stayed strong at 7.06-7.16.

The apparent sequence plateau mixed two different protocols. The 1,000
pure-language validation rows are scored semantically by the preregistered
`first_nonempty_completion_line_v1` calibration and pass at 100%, but joint
teacher forcing required an exact continuation/newline/EOS token sequence and
reported those rows at 0% strict sequence accuracy. Replacing only that known
diagnostic mismatch gives a protocol-aware lower bound of about 83.6%, close
to the Stage-9 soft result, while leaving every communication-dependent class
strict. Pure-language rows also have an exact all-closed receiver bypass, so
they cannot train the adapters and consumed 20% of the original optimizer
stream without providing adapter gradients.

A paired FP32 panel across epoch-1 and epoch-2 checkpoints found 19 changed
predictions, five net additional correct tokens, and one net additional exact
sequence. The saturated explicit-math, language-dependent-math, and
multi-sequential strata stayed correct; remaining error and movement were
concentrated in exact-string and multi-parallel rows. The old checkpoint score
incorrectly added `0.05 * causal_gap`, allowing noise from only 408 causal
examples to outweigh full-panel token/loss gains. It therefore kept epoch 1 as
"best" even while the adapters improved.

The continuation contract consequently:

- validates all six classes but excludes pure-language rows from optimization;
- weights exact-string and multi-parallel task loss above replay classes;
- starts a fresh optimizer at 3e-7 (floor 5e-8), rather than resuming stale
  optimizer, scheduler, best-score, and patience state;
- ranks only checkpoints that preserve protocol-aware accuracy, protected
  solved classes, and causal gap >=5;
- reports strict token, sequence, and GPT loss per task class;
- treats causal gap as an acceptance guard rather than a ranking bonus; and
- limits the continuation to four epochs with patience 2 before any router
  calibration is allowed.

The guarded source selector considered retained epochs 2-4 and selected epoch
3 (`0df17b1edacf81320616b523cf77297524fddcfb9496674cbb9593b165201350`)
with proxy score 0.811316. Epoch 4 was not selected because its small token/loss
gain came with a strict sequence decline (proxy 0.810776). The continuation was
launched as isolated PID 24696; its source selection, source hash, contract,
and logs are preserved separately from the completed first adapter run.

For V2, carry forward the protocol separation and per-class reporting. Never
optimize on rows whose hard-closed path bypasses all trainable collaboration
components, and never let a small causal panel dominate checkpoint rank after
it has passed a preregistered causal floor.

## Remaining tests

- Full 5,000-example validation after one routing-only Stage-10 epoch.
- Gate-probability margin distribution around the 0.5 threshold.
- Zero-message receiver versus receiver-disabled numerical identity.
- Separate hard-halt zero-update panel after wake routing passes.
- Final autoregressive Stage-11 causal suite with actual execution counts.

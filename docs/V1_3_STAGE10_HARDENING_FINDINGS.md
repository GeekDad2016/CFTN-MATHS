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

## Route-schedule falsification and fusion recovery (2026-08-22)

The continuation selected its own epoch-2 checkpoint
(`236d3b4b71b595cf99f6babf2d17f090823f5a2b716e32c671487a471becdb99`).
It retained 100% strict accuracy on explicit math, language-dependent math,
and multi-sequential rows, reached 66.7% on exact string, but only 3.4% on
multi-parallel composition. This isolated the remaining failure to combining
simultaneous specialist returns.

An exhaustive hard route-schedule sweep evaluated all 84 one-, two-, and
three-round schedules on 500 multi-parallel validation rows. The intended
single `both` call scored 3.0% exact sequence accuracy, 63.66% token accuracy,
and GPT loss 1.4238. The best schedule, `both > closed > math`, reached only
3.4% exact sequence accuracy while reducing token accuracy to 55.30% and
raising GPT loss to 2.0751. Its 0.4-point exact gain costs another specialist
call and is not material. Repeated `both`, math-first, and string-first
schedules also failed. Therefore neither supervised route calibration nor
gate-only RL has a useful action policy to discover with the current message
consumer.

The next recovery changes the message consumer, not the trained towers or
router:

- add specialist, round, and slot type embeddings plus self-attention over
  returned message tokens;
- apply the learned transformation as a residual with a zero-initialized final
  projection, making the preserved checkpoint an exact identity at step zero;
- train only this message-fusion module and the GPT receiver adapters;
- keep specialists, request/return bridges, specialist receivers, wake gates,
  and halt gate frozen;
- use oracle hard two-round execution on a deterministic 18,000-row subset
  weighted toward multi-parallel and exact-string cases, with protected replay;
- validate all six classes and retain the zero-update source as an eligible
  epoch-0 candidate; and
- promote only a checkpoint that improves the focus-class selection score
  while preserving protocol-aware accuracy, solved classes, and causal gap
  >=5.

The real epoch-2 checkpoint passed a batch-16 BF16 forward/backward probe on
the RTX 4070. Loss was finite, only `message_fusion` and `gpt_receivers` had
trainable parameters, and no frozen component received a gradient. Peak GPU
allocation was 2.75 GiB in the probe. These findings and the route-sweep report
must be carried into V2 before its composition/fusion stage; adding RL to V2
routing before establishing a rewarding hard action is specifically ruled
out.

## Fusion recovery outcome (2026-08-22)

The fusion recovery completed six epochs and selected its accepted epoch-4
checkpoint
(`231566e1d17dca7a35bd7028c7af38bb9f450c38ff9f727fa7d278dcbb8cd790`).
Only `message_fusion` and `gpt_receivers` were optimized. The specialists,
request and return bridges, specialist receivers, wake gates, and halt gate
remained frozen. The selected checkpoint preserved the protected solved
classes, protocol-aware lower bound, and causal gap and improved the registered
focus-class selection score.

Fusion was necessary but not sufficient. It improved the simultaneous-message
consumer and provided the accepted source for the next experiment, but native
free-running exact composition still required an independently gated output
path.

## Answer-bus recovery and native-transfer failure (2026-08-23)

A typed lossless byte answer bus and pointer-copy answer composer were trained
from the accepted fusion checkpoint. The clean registered-bus phase selected
checkpoint
`cd5ef65a948607f53629bfd0e3837f3e6966c0497fbccfe15e11bb8f270f9ccf`
with 100% generated answer-composer accuracy and all clean-bus acceptance gates
passing.

The subsequent 748-example native evaluation failed despite 96.39% answer-bus
format validity:

- exact-string task accuracy: 16.67%;
- explicit math: 23.44%;
- language-dependent math: 34.38%;
- multi-parallel: 0%; and
- multi-sequential: 10.16%.

This established a clean-versus-native bus distribution mismatch. Format
validity was not evidence that native specialist outputs transferred through
the learned composer. A controlled native/noisy continuation preserved the
failed report and source checkpoint but also failed acceptance; no continuation
checkpoint was promoted.

## Typed-request and deterministic-composition resolution (2026-08-24)

Direct tower diagnostics then separated request construction from tower
capability. Complete typed native requests produced the expected registered
specialist payloads. The final accepted runtime therefore removed two learned
exactness bottlenecks:

1. A learned dispatcher predicts only one finite intent and confidence.
2. A constrained compiler copies operands from immutable public-prompt spans
   into typed native requests; it never generates operands or reads oracle
   metadata.
3. Specialist payloads are carried through typed result references.
4. Final exact results are composed deterministically rather than through the
   failed learned answer composer.
5. The legacy latent wake/halt request path is bypassed for accepted inference.

The learned dispatcher checkpoint
`82c166ecada2d5aa148bdd63b6d205ad976adc44493b2e159d40aea859b7f6d0`
passed 5,500 validation examples, 5,500 held-out paraphrases, and 4,500
independent semantic/unsupported examples at 100% accuracy and 100% coverage
with confidence threshold 0.9.

The final public-prompt native panel passed all gates over 748 examples with no
oracle metadata visible to runtime: 100% dispatch-plan validity, dispatch
completion, specialist-payload validity, and answer-bus validity; 99.33% exact
string, 100% explicit math, 100% language-dependent math, 99.48%
multi-parallel, and 100% multi-sequential accuracy.

The machine-readable final ledger is
`evidence/v1_3_recovery_final_evidence.json`. The original registered Stage 11
and Stage 12 were not run and are not retroactively claimed as complete.

## Tests still outside the accepted claim

- A separately calibrated learned hard-halt runtime was not completed. The
  accepted typed dispatcher provides the active conditional gate instead.
- The original preregistered Stage-11 dense-versus-conditional compute table was
  not produced.
- Open-world dispatch beyond the registered task grammar remains untested.

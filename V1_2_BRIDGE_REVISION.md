# CFTN-Text V1.2: conditional communication revision

Status: active experiment. The sealed V1.1 pipeline completed all 15 stages,
the prerequisite audit passed, and V1.2 training started on 2026-08-10.

## Purpose

V1.2 is the missing experiment between the narrow V1.1 proof and the broader
V2 specialist curriculum. It does not add a larger language model, a larger
math tower, expert routing, wake gates, or end-to-end tower fine-tuning. It asks
one focused question:

> Can a contextual corpus-callosum bridge transmit prompt-specific information
> when the math specialist needs it, while remaining residual-neutral when the
> specialist already has enough information?

This is the behaviour required by the intended CFTN system. The generalist
maintains language context; the specialist receives only the information and
native input needed for its narrow computation; the return bridge places the
computed result back into the generalist's evolving state.

## Evidence that motivates the revision

V1.1 established a scoped mechanism-level success:

- In complementary private view, familiar full-CFTN accuracy was 98.65% while
  GPT alone was 0% and the math-side readout with GPT-to-math disabled was
  0.30%.
- On the compositional split, full CFTN reached 87.00%; the correct math-side
  readout reached 86.80%, GPT-to-math disabled reached 0.45%, and shuffled
  GPT-to-math reached 15.00%.
- Aggregate complementary-view accuracy was 42.15%, versus 0.23% for the
  strongest isolated arm and 0.33% with both messages shuffled.
- Both directional causal gains and the correct-versus-shuffled gate passed
  with positive 95% intervals.

This demonstrates that compact bidirectional messages can combine information
that neither frozen tower possesses alone.

V1.1 also exposed four limitations:

1. In shared view, GPT-to-math was redundant and damaged an already-capable
   specialist. The gate had never been trained on a justified negative case in
   which communication should be neutral.
2. Held-out language and numerical extrapolation remained weak. On held-out
   language, correct GPT-to-math exceeded shuffled GPT-to-math by only 1.15
   points, so unseen wording did not produce a sufficiently content-specific
   message.
3. Teacher-forced validation overstated greedy-generation capability.
4. Fixed-open one-way and bidirectional controls collapsed after their early
   best checkpoints. Always-on residual communication is not a safe substitute
   for contextual gates.

The formal V1.1 collaboration claim remains failed because broad
counterfactual and donor-follow targets were not met. V1.2 must preserve that
sealed result rather than reinterpret its thresholds after seeing the test
data.

## Input contract

V1.2 trains on two deterministic views of each training and validation record:

1. **Bridge required / complementary:** GPT receives the semantic operation and
   role mapping; the math tower receives opaque numeric slots. Correct
   GPT-to-math communication is necessary.
2. **Bridge redundant / shared:** both towers receive the complete problem. The
   frozen math tower can solve independently, so GPT-to-math must not reduce its
   result.

The two views use the same target and are kept in the same training curriculum.
They are training/validation data only. The V1.1 test, held-out-language,
extrapolation, answer-extrapolation and compositional results remain sealed.

V1.2 deliberately does not add non-math tasks yet. A later gate-calibration
stage should add tasks requiring zero, one and multiple specialists before wake
gates or conditional compute are introduced.

## Frozen and trainable components

Start from the successful V1.1 contextual bidirectional best checkpoint.

Frozen:

- GPT-2 language tower;
- math tower;
- math-to-GPT message bridge and GPT receiver modules.

Trainable:

- GPT-to-math message bridge;
- math receiver modules;
- their contextual sender and receiver gates.

Freezing the proven return path makes the experiment directional and causal:
any improvement or regression is attributable to the revised GPT-to-math path.
The saved checkpoint still contains the complete bidirectional state so the
standard shared and complementary evaluators can load it.

## Training objective

For a correct bridge pass, let `L_task` be the existing math, GPT and optional
answer-head loss. For redundant shared examples, calculate a frozen
GPT-to-math-disabled baseline and preserve its correct math-token distribution.
For required complementary examples, compare the correct message with disabled
and shuffled controls.

The revision objective is:

```text
L = L_task
  + lambda_preserve * L_specialist_preservation
  + lambda_contrast * L_required_message_margin
  + lambda_neutral * L_redundant_gate
```

- `L_specialist_preservation` is a masked KL loss against the detached
  GPT-to-math-disabled math logits, applied only to shared examples whose frozen
  baseline sequence is correct.
- `L_required_message_margin` requires the correct complementary message to
  obtain lower math loss than both disabled and shuffled controls. Controls are
  detached so the model cannot satisfy the objective merely by making corrupted
  messages arbitrarily destructive.
- `L_redundant_gate` softly penalizes GPT-to-math sender and math-receiver gate
  activation only on explicitly redundant shared examples.

The gate penalty is not a global sparsity objective. Required examples are not
given an arbitrary open target; task and causal-margin losses must make useful
communication emerge.

Initial settings:

- paired required/redundant sampling: 50/50;
- bridge learning rate: 2e-5;
- contextual-gate learning rate: 5e-6;
- preservation weight: 2.0;
- required-message contrast weight: 0.5;
- detached loss margin: 0.25 nats;
- redundant-gate weight: 0.05;
- gradient clipping: 1.0;
- BF16 on CUDA;
- retain the latest three checkpoints and a separate best checkpoint.

These are preregistered starting values, not claims that the gate must adopt a
particular absolute activation.

## Validation and checkpoint selection

Every epoch runs teacher-forced diagnostics and a fixed greedy-generation panel
containing both required and redundant examples. The generation panel evaluates
identical records under:

- correct bidirectional messages;
- GPT-to-math disabled;
- GPT-to-math shuffled;
- math-to-GPT disabled;
- both bridges disabled.

Checkpoint selection is generation-led. It rewards required-view joint accuracy
and direction-specific causal gains, while strongly penalizing shared-view math
regression. Teacher-forced loss remains a stability diagnostic rather than the
primary capability score.

The V1.2 training acceptance gate requires all of the following on fixed
validation records:

- required-view full CFTN exceeds the stronger isolated arm by at least 10
  points;
- required-view correct GPT-to-math exceeds shuffled GPT-to-math by at least 2
  points;
- disabling math-to-GPT removes at least 10 points from the final GPT answer;
- redundant-view full math accuracy is no more than 2 points below the
  GPT-to-math-disabled frozen baseline;
- mean GPT-to-math gate activation is higher for required than redundant
  examples by at least 0.05;
- no collapse guard triggers.

Final sealed evaluation must additionally run the existing shared-view and
complementary-view ablations on exactly the same records used by V1.1. V1.2 is
accepted only if it improves communication safety without erasing V1.1's
familiar and compositional complementary-view success.

## Automatic pipeline

The V1.2 runner is resumable and refuses to start until V1.1 reports terminal
completion. Its stages are:

1. audit the immutable V1.1 data, source math checkpoint, source contextual
   bridge checkpoint and completed V1.1 evidence;
2. train the conditional GPT-to-math revision while preserving math-to-GPT;
3. evaluate shared-view no-harm controls;
4. evaluate complementary-view causal synergy using the existing sealed
   benchmark;
5. compare V1.2 against V1.1 and assemble a final revision report.

No stage automatically modifies V1.1 artifacts.

Stage 5 writes `v1_2_final_report.json` and a human-readable
`V1_2_EXPERIMENT_RESULTS.md` in both the V1.2 artifact directory and repository
root. The document records immutable provenance, what passed, what failed,
measured evidence, diagnostic hypotheses, limitations, and recommended fixes
or the next experiment. A V1.2 pass advances to the preregistered V1.3
multi-specialist wake-gate experiment. A failure advances to a targeted V1.2.x
repair instead.

## Relationship to V1.3, V2 and the multi-tower roadmap

If V1.2 passes, V1.3 is the immediate next mechanism experiment. It adds a
second exact-string specialist, independent wake gates, mixed tasks requiring
zero, one, several or all specialists, and bounded recurrent callosal rounds.
Its preregistered design is in `V1_3_EXPERIMENT_PLAN.md`.

V2 remains the broad specialist-capability experiment: more problem families,
larger numerical ranges, paraphrases, extrapolation, GSM8K, GSM-Symbolic and the
DeepMind Mathematics Dataset. Broad scaling remains separate from the V1.3
conditional-participation proof because more specialist knowledge does not
solve unsafe communication or wake-gate control.

After V1.2 passes, V2 should retain the same division of labour:

- the language tower receives the natural problem context;
- the specialist receives a narrow native workspace rather than being forced
  to become another broad language model;
- GPT-to-specialist messages carry interpreted operation, roles and relevant
  spans;
- specialist-to-GPT messages carry exact results and compact supporting state.

The current V2 generated benchmark externally prepares language and numeric
private views. That is a controlled causal test, not yet proof that GPT can form
the complete expert request from an untouched natural prompt. A later V2
natural-interface arm should give only GPT the raw prompt and initialize the
specialist from neutral scratch/message tokens.

Wake gates, conditional specialist execution and iterative callosal reasoning
remain later milestones. They are introduced only after dense message-gated
communication is accurate, causal, context-sensitive and non-destructive.

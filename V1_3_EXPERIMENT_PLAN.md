# CFTN-Text V1.3: wake-gated recurrent multi-specialist experiment

Status: preregistered successor. Start only if V1.2 passes its sealed
conditional-communication gate. If V1.2 fails, run a targeted V1.2.x bridge
repair first and record that decision in `V1_2_EXPERIMENT_RESULTS.md`.

Implementation status: isolated and executable. The implementation lives in
`config/v1_3_multi_specialist.yaml`, `cftn_text/v1_3_*`, and
`tools/*v1_3*`. `tools.wait_then_run_v1_3` is the sole automatic continuation:
it verifies the V1.2 revision, final gates, pipeline state, report hash, and
checkpoint hash before starting the 12-stage V1.3 pipeline. The terminal check
also requires the exact five-stage V1.2 completion ledger, no active stage, and
the sealed evidence stage itself. Direct stage entry
points repeat the same audit. Until a passing sealed report exists, the guard
must fail closed and create no V1.3 training artifacts.

## Research question

V1.2 asks whether a bridge can communicate when useful without damaging a
specialist that already has enough information. V1.3 asks the next question:

> Can one persistent language workspace wake zero, one, or several independent
> specialists only when their capabilities are needed, exchange causally useful
> information over one or more latent reasoning rounds, and preserve answer
> quality while reducing specialist computation?

This is conditional participation, not mixture-of-experts routing. GPT remains
the persistent problem context and final-response system. Specialists do not
compete to replace GPT, receive the whole example as a routed batch, or produce
independent answers for weighted voting. Each wake decision is independent;
several specialists may be active in the same round.

## Scope and preconditions

The first V1.3 system uses the smallest honest multi-specialist configuration:

1. the frozen V1.2 language tower and its proven communication mechanisms;
2. the frozen arithmetic/algebra specialist proven in V1.1/V1.2;
3. a new exact-string specialist trained independently before integration.

With two specialists, `all specialists` means both math and string. More towers
are added only after this experiment passes.

V1.3 has three hard preconditions:

- V1.2 passes conditional message utility and shared-view no-harm;
- each specialist passes its own sealed native-capability test;
- the frozen GPT baseline passes the selected no-specialist language task
  family before mixed-system training begins.

The native-specialist precondition is task matched, not merely generic. Before
any bridge integration begins, every required math and string operation in the
primary `joint_test` contract is also presented to its specialist through that
specialist's proven native interface. Every specialist/task-class cell must
reach at least 95% exact accuracy, and at least 95% of specialist-requiring
primary examples must be solvable by all specialists they require. Failure
stops the pipeline before bridge training.

If GPT-2 cannot reach 90% on a clean calibration set of no-specialist tasks,
those tasks cannot measure wake-gate precision. Either choose a task family the
frozen generalist genuinely knows, or preregister a stronger dense generalist
as a separate architecture revision. Do not select individual sealed test
examples based on GPT success.

The initial Stage-3 attempt exposed an interface confound: frozen base GPT-2
repeated an instruction-style prompt instead of emitting XML answer tags. The
registered repair keeps the 90% semantic gate unchanged and uses GPT-2 as the
completion model it was pretrained to be. Pure-language controls are rendered
as key-value records ending in `Requested archival label:`; all other GPT
prompts end in `Exact result:`. The answer contract is the first non-empty
completion line. Training targets the raw answer followed by a newline, while
tag accuracy remains a non-gating formatting diagnostic. Calibration reports
semantic accuracy, valid-field rate, strict tag accuracy, and prompt-copy rate
separately. This repair must regenerate the hashed V1.3 data contract and may
not lower the threshold or count the prompt itself as generated output.

## Specialist capabilities

### Math specialist

Reuse the strongest frozen math checkpoint available after V1.2. Its native
capability report must state exact accuracy by operation family, wording,
number range, and compositional depth. V1.3 does not assume knowledge outside
that report.

### Exact-string specialist

Train a small byte-level transformer on deterministic tasks:

- string length;
- character frequency;
- character indexing and position queries;
- reversal;
- substring matching;
- simple substitutions;
- two or more constraints over unseen strings.

The target definition must distinguish UTF-8 bytes, Unicode code points, and
grapheme clusters. The first capacity experiment should use ASCII; Unicode is a
separate sealed extension.

Use approximately 100,000 to 250,000 unique generated examples, with held-out
strings, lengths, operations, natural-language paraphrases, and compositions.
Scale only if the held-out learning curve is still improving.

## Natural interface

The untouched user prompt is given only to GPT. A specialist starts from a
small learned neutral workspace containing its identity and control tokens. It
does not receive a separately prepared full natural-language prompt.

When a wake gate activates specialist `i`, GPT sends a bounded latent request
`m(G->i)`. That message must carry the relevant operation, operands, semantic
roles, constraints, or text span. The specialist computes in its native
representation and returns `m(i->G)` containing an exact result and compact
supporting state.

For diagnostic purposes, V1.3 may include an oracle-native-input arm, but the
main result must use the GPT-only raw-prompt interface. This prevents an
external parser from silently performing the central communication task.
Oracle-native inputs are used only to establish whether the receiving
specialist already possesses the required operation; they are never supplied
to the main CFTN arm.

## Mixed task curriculum

Generate a balanced mixed curriculum in which the required specialist set is
known by construction. A starting distribution is:

| Task class | Share | Required specialists | Example purpose |
| --- | ---: | --- | --- |
| Pure language | 20% | none | GPT answers from stated context; all specialists should sleep |
| Explicit pure math | 20% | math | GPT receives a symbolic or direct arithmetic question and relays it to math |
| Exact string | 20% | string | GPT interprets the instruction and relays the span and operation to string |
| Language-dependent math | 20% | math | GPT must resolve wording, roles, references, or distractors before math can compute |
| Multi-specialist | 20% | math and string | Both towers are necessary, in parallel or sequentially |

Half of the multi-specialist examples should be parallel: both outputs can be
computed from the original prompt. Half should be sequential: one specialist's
result becomes an operand or constraint for the other. For example, count the
`r` characters in an unseen string and then use that count in an equation.
Sequential examples are the cleanest test that another callosal round adds
capability rather than merely repeating the first pass.

Every class needs minimally different counterfactual pairs. Change the
operation while preserving vocabulary, change one operand, or change whether a
second specialist is required. This prevents wake gates from succeeding via
keywords such as `equation` or `letters` alone.

Use disjoint generator seeds and structural templates for train, validation,
and sealed test. Report familiar, unseen-paraphrase, numerical/string
extrapolation, distractor, counterfactual, and unseen-composition splits.

Only `joint_test` is the primary acceptance split. Its bridge, wake, synergy,
and recurrence claims are computed on examples whose required specialists
independently solved the task-matched oracle-native controls. This conditioning
cannot hide a weak specialist: primary competence coverage itself must be at
least 95%. Held-out paraphrase, extrapolation, counterfactual, and unseen
composition are separately reported non-gating diagnostics. They may identify
knowledge boundaries, but a failed diagnostic cannot be attributed to a
bridge when the required specialist failed its oracle-native control.

## Runtime architecture

Each specialist has two independent controls:

1. a cheap **wake gate** predicts whether that specialist should execute in the
   current round;
2. continuous **message gates** control the content and strength of both bridge
   directions after it is awake.

The initial runtime uses at most three callosal rounds:

```text
round 0: GPT encodes the raw prompt
round 1: wake relevant specialists -> specialists compute in parallel
         -> return messages update GPT state
round 2: GPT consolidates results and may wake specialists again
round 3: optional final specialist refinement -> GPT produces the answer
```

A learned halt gate may end after any round. Parallel specialists execute
concurrently. A dependent specialist executes in a later round only after the
required upstream result has updated GPT state.

The primary reasoning mechanism is bounded latent recurrence, not unconstrained
textual chain-of-thought. Expert requests, wake decisions, result receipts, and
final answers remain measurable. An optional short rationale may be generated,
but exact task correctness and causal message dependence are primary.

## Training sequence

1. Train and seal each specialist independently.
2. Freeze GPT and all specialists.
3. Train each new bidirectional bridge on small single-specialist capacity
   sets, with correct, closed, and shuffled controls.
4. Run all specialists densely and train continuous message gates on the mixed
   curriculum. Carry forward V1.2 preservation, bridge-dropout, no-harm, and
   causal-message-margin losses.
5. Add recurrent callosal rounds while all specialist computation remains
   dense and differentiable.
6. Train cheap wake gates using both required-set labels and causal utility.
   A wake label alone is insufficient: disabling the labelled expert must
   damage the examples that claim to require it.
7. Calibrate gate probabilities, then harden wake decisions gradually.
8. Add a modest compute-cost objective only after the hard-wake model matches
   the proven dense model within two accuracy points.
9. Seal the checkpoint and run every ablation on identical examples.

The loss family is:

```text
L = L_task
  + lambda_message * L_causal_message
  + lambda_preserve * L_no_harm
  + lambda_wake * L_required_set
  + lambda_utility * L_causal_wake_utility
  + lambda_halt * L_round_halt
  + lambda_compute * L_active_compute
```

`L_active_compute` starts at zero and is enabled only during hardening. This
prevents the model from learning cheap silence before it learns cooperation.

## Matched evaluation arms

Run all arms on the same sealed examples:

- GPT alone;
- each specialist alone;
- dense CFTN with every specialist computed and learned message gates;
- learned wake-gated CFTN;
- oracle wake-set CFTN;
- all wake gates closed;
- each specialist independently disabled;
- each bridge direction independently disabled;
- messages shuffled between examples;
- the wrong specialist forced awake;
- all specialists always awake with fixed-open messages;
- one callosal round only;
- simple GPT-to-tool serial pipelines using the same specialists.

For sequential examples, also swap first-round specialist returns between
matched pairs and verify that downstream computation follows the donor result
only when the swapped message is semantically relevant.

## Metrics

Report:

- exact final-answer accuracy and valid-answer rate;
- synergy over the strongest individual tower and serial pipeline;
- wake precision, recall, F1, exact required-set accuracy, and calibration;
- false-wake rate on pure-language tasks;
- irrelevant-specialist activation on single-specialist tasks;
- correct, disabled, shuffled, and swapped message gaps;
- no-harm regression on tasks already solved without a given message;
- accuracy by zero, one, several, and all required specialists;
- accuracy by parallel versus sequential multi-specialist composition;
- halt-round distribution and the gain from additional rounds;
- measured specialist FLOPs, active parameters, latency, energy where
  available, and peak memory;
- 95% paired confidence intervals for all central differences.

## Preregistered acceptance criteria

V1.3 passes only if all central gates pass:

1. Every specialist reaches at least 95% exact accuracy on its sealed native
   familiar test and on every task-matched oracle-native cell used by the
   primary benchmark. At least 95% of specialist-requiring primary examples
   must remain competence-supported. Generalization splits are reported
   separately and are non-gating.
2. Pure-language performance is no more than two points below GPT alone and
   wakes any specialist on at most 5% of examples.
3. Required-specialist wake recall is at least 95%, wake precision at least
   90%, and exact required-set accuracy at least 90%.
4. On joint tasks, learned-wake CFTN exceeds the strongest individual tower by
   at least 10 points with a paired 95% interval above zero.
5. Closing or shuffling each required bridge removes at least 10 points on the
   task class that needs it; irrelevant bridge ablation changes accuracy by no
   more than two points.
6. Learned hard wake is within two points of dense CFTN while reducing measured
   specialist computation by at least 30% over the balanced mixed workload.
7. Sequential multi-specialist accuracy is at least 80%, and limiting the
   model to one callosal round causes at least a 10-point loss on the
   preregistered sequential subset.
8. Learned wake/message gates outperform fixed-open communication and the
   simple serial pipeline on the central joint benchmark.
9. No collapse, always-open, always-closed, single-expert monopoly, or answer
   copying diagnostic triggers.

Failure of any criterion remains useful evidence. Do not change thresholds
after reading the sealed test.

## Result-document contract

The completed experiment writes both machine-readable JSON and
`V1_3_EXPERIMENT_RESULTS.md`. The human-readable result must contain:

- immutable data/config/checkpoint hashes and run links;
- the exact architecture and training sequence used;
- what passed and what failed;
- evidence for every claim and confidence interval;
- likely causes clearly labelled as hypotheses rather than observations;
- limitations and knowledge boundaries of every tower;
- recommended fixes or the next experiment;
- enough context that a later review does not reinterpret an old result.

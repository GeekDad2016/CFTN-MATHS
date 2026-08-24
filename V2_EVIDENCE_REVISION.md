# CFTN-Text V2 scaled multi-specialist revision

V2 now trains the same mechanism claims as V1.3 with larger specialists and
larger datasets. It does not import a V1.3 collaboration checkpoint and does
not require V1.2 or V1.3 to pass before training. Earlier reports are retained
only as historical provenance.

## What V2 trains

- Frozen GPT-2 remains the general-language workspace and answers requests
  that need no specialist. Specialist-backed answers use typed deterministic
  result composition rather than asking GPT to reproduce exact payload bytes.
- A broad byte-level math tower is trained from scratch on the fixed 400,000
  example curriculum and selected by validation-only greedy generation.
- A larger byte-level exact-string tower is trained from scratch on 200,000
  examples.
- Fresh GPT-to-specialist and specialist-to-GPT bridges and receivers are
  trained while both native towers remain frozen.
- Independent sigmoid wake gates can activate no specialist, either
  specialist, or both. They are not softmax, top-k, or winner-take-all routing.
- Three callosal rounds permit sequential cooperation and a learned halt gate.
- Hard mode physically skips inactive specialist execution in the sealed
  evaluation.
- A separately trained value-invariant dispatcher selects a finite typed call
  graph from the public user prompt. It routes broad math, one-specialist,
  parallel, and both sequential dependency orders without task labels.

The native specialists see neutral workspaces during joint training. GPT must
communicate task-relevant information to a required specialist, and return
messages must carry information that GPT can use in its answer. Pure-language
examples require no specialist; explicit math and exact-string examples
require one; language-dependent and multi-specialist examples exercise
directional and recurrent cooperation.

## Evidence incorporated from V1.3

The first V1.3 hardening attempt demonstrated that a high learning-rate restart
and continued bridge/receiver updates could destroy excellent soft routing,
producing an always-open gate collapse. V2 therefore treats hardening as a
separate transition with these executable safeguards:

1. select the best supervised-soft-wake checkpoint;
2. evaluate that exact checkpoint in hard mode with zero optimizer updates;
3. record soft and hard metrics on the same validation panel;
4. freeze GPT, specialists, bridges, receivers, and the halt gate;
5. train only wake gates with required-set BCE, while hard halt remains
   disabled until a separate calibration experiment;
6. use no warmup and cap gate LR at `5e-7` (floor `2.5e-7`);
7. reject checkpoints that violate false-wake, exact-set, precision, recall,
   baseline-regression, always-open, or always-closed guards;
8. prevent any ineligible checkpoint from becoming the selected Stage 16
   checkpoint or entering the final causal evaluation.

The soft-to-hard baseline is diagnostic rather than a training gate inherited
from V1.3. V2 measures its own thresholding loss because its towers, bridges,
receivers, and gates are newly trained.

The later V1.3 native confirmation also showed that high answer-bus format
validity does not guarantee task transfer: a learned answer decoder trained on
clean buses failed on native tower outputs. V2 therefore carries forward the
confirmed repair as an executable contract:

1. the learned component predicts only a finite intent/call graph;
2. quoted values, numeric operands, and broad-math prompts are immutable source
   spans, never generated arguments;
3. each typed call is rendered into the target specialist's independently
   validated native interface;
4. result dependencies are explicit and may only refer to an earlier round;
5. exact specialist payloads are composed deterministically by typed result
   references;
6. unsupported, structurally invalid, or low-confidence requests fail closed;
7. dispatcher validation covers the registered grammar, held-out paraphrases,
   broad-math language, and independent semantic/unsupported controls;
8. a native end-to-end panel runs with only public prompt fields and must prove
   that no task, wake, target, or oracle-specialist metadata reached runtime.

## Reserved third specialist

The registry has capacity for three specialist roles. `math` and `string` are
active. `extension_1` is explicitly `reserved_inactive` while its capability
and dataset are being chosen. It is absent from wake targets, model modules,
optimizers, checkpoints, and compute, so a placeholder cannot distort this
run. Activating it requires a new sealed config that defines:

- native inputs, targets, tokenizer, tower builder, and checkpoint contract;
- standalone familiar and task-matched competence gates;
- zero-, one-, two-, and three-specialist joint examples;
- direction-disabled, shuffled-message, no-harm, recurrence, and compute arms;
- backward-regression tests for math and exact-string behavior.

## Ordered pipeline

1. prepare and hash broad-math data;
2. train broad math for at least 60 and at most 100 epochs, stopping only after
   10 validation epochs without improvement once the minimum is reached;
3. select the math checkpoint by validation greedy generation;
4. evaluate sealed standalone math generation;
5. record whether later math-data scaling is justified (never auto-scale);
6. prepare and hash multi-specialist/string data;
7. train and seal the learned constrained dispatcher;
8. calibrate frozen GPT on pure-language prompts;
9. train the larger exact-string specialist for at most 30 epochs;
10. seal native specialist competence and task-matched coverage;
11. train one-round single-specialist capacity for 8 epochs;
12. train dense mixed messages for 12 epochs;
13. train dense recurrent cooperation for 12 epochs;
14. train supervised soft wakes for 10 epochs;
15. evaluate the selected soft checkpoint in hard mode with zero updates;
16. harden only wake gates for at most 10 epochs, with the halt gate frozen;
17. run native typed dispatch and deterministic composition with oracle metadata
    removed from runtime records;
18. run the sealed causal, no-harm, recurrence, routing, and compute suite;
19. assemble `v2_final_report.json` and `V2_EXPERIMENT_RESULTS.md`.

## Acceptance and claim boundary

The final claim requires native competence, pure-language no-harm, low false
wake, wake precision/recall, exact per-round required sets, positive
multi-specialist synergy, causal loss under required direction/message
ablations, low irrelevant-specialist effect, hard-vs-dense preservation,
conditional-compute reduction, sequential accuracy, and multi-round gain.
It additionally requires passing dispatcher accuracy/coverage on every sealed
panel and passing native end-to-end typed dispatch, request completion, answer
bus validity, task accuracy, no-oracle, and deterministic-composition gates.

Competence-conditioned scores prevent missing specialist knowledge from being
misreported as a bridge failure. Diagnostic paraphrase, extrapolation,
counterfactual, and unseen-composition splits remain non-gating. A pass would
support controlled two-specialist cooperation with conditional execution; it
would not establish broad intelligence, arbitrary plug-and-play experts, or a
completed three-specialist system.

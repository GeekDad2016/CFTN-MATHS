# CFTN-Text V2 scaled multi-specialist revision

V2 now trains the same mechanism claims as V1.3 with larger specialists and
larger datasets. It does not import a V1.3 collaboration checkpoint and does
not require V1.2 or V1.3 to pass before training. Earlier reports are retained
only as historical provenance.

## What V2 trains

- The dense `Qwen/Qwen3-4B-Instruct-2507` checkpoint remains frozen as the
  general-language coordinator and answers requests that need no specialist.
  It is pinned to full revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`; startup rejects MoE or
  shape-incompatible substitutes. Specialist-backed answers use typed
  deterministic result composition rather than asking Qwen to reproduce exact
  payload bytes.
- A broad byte-level math tower is trained from scratch on the fixed 400,000
  example curriculum and selected by validation-only greedy generation.
- A larger byte-level exact-string tower is trained from scratch on 200,000
  examples.
- Fresh Qwen-to-specialist and specialist-to-Qwen bridges and receivers are
  trained while both native towers remain frozen.
- Independent sigmoid wake gates can activate no specialist, either
  specialist, or both. They are not softmax, top-k, or winner-take-all routing.
- Three callosal rounds permit sequential cooperation and a learned halt gate.
- Hard mode physically skips inactive specialist execution in the sealed
  evaluation.
- A separately trained 5,025,996-parameter hierarchical dispatcher combines
  the frozen Qwen semantic prepass with byte-level structural validation. Its
  heads predict delegation, a multi-label tower set, dependency rounds, and a
  finite typed call graph from the public user prompt. Exact arguments remain
  copied source spans, and inactive slots are impossible to select.

The native specialists see neutral workspaces during joint training. Qwen must
communicate task-relevant information to a required specialist, and return
messages must carry information that Qwen can use in its answer. Pure-language
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
4. freeze Qwen, specialists, bridges, receivers, and the halt gate;
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

## Twelve-slot specialist registry

The dispatcher and registry have stable capacity for twelve specialist roles.
`math` and `string` are active. The ten named slots `code`, `formal_logic`,
`science`, `retrieval`, `long_context`, `multilingual`, `tool_use`,
`structured_data`, `information_extraction`, and `commonsense` are explicitly
`reserved_inactive`. They are masked out of dispatcher loss and selection and
are absent from wake targets, tower modules, optimizers, checkpoints, and
compute, so placeholders cannot distort this run. Activating any one requires
a new sealed config that defines:

- native inputs, targets, tokenizer, tower builder, and checkpoint contract;
- standalone familiar and task-matched competence gates;
- zero-, one-, and multi-specialist joint examples, including that tower's
  parallel and sequential dependencies;
- direction-disabled, shuffled-message, no-harm, recurrence, and compute arms;
- backward-regression tests for math and exact-string behavior.

## Current parameter footprint

The configured V2 architecture was instantiated and counted directly on
2026-08-24. Counts are unique model parameters, not checkpoint bytes or the
sum of parameters activated on every request.

For comparison, the confirmed V1.3 model plus its learned dispatcher contains
155,369,206 parameters (0.155B). The new dense-Qwen V2 target skeleton contains
4,089,525,858 parameters (4.090B). This is the coordinator plus the two active
towers, all current integration modules, and the hierarchical dispatcher; the
ten reserved tower slots allocate no tower parameters.

| Component | Parameters |
| --- | ---: |
| Frozen dense Qwen3-4B coordinator | 4,022,468,096 |
| Broad-math tower | 18,040,449 |
| Exact-string tower | 16,070,273 |
| Hierarchical semantic/structural dispatcher | 5,025,996 |
| Qwen receivers plus retained bridges, fusion, gates, halt, and answer composer | 27,921,044 |
| **Current resident V2 target skeleton** | **4,089,525,858 (4.090B)** |

The accepted typed runtime core will be Qwen, both native specialists, and the
dispatcher once V2 has trained and passed; this document does not claim that
the new target has done so yet. Retained integration parameters support causal
diagnostics and checkpoint compatibility, but deterministic typed composition
does not depend on the learned answer composer or latent request path.
Conditional execution means active parameters per request are lower than
resident parameters whenever only one specialist, or no specialist, is used.

## Capability-tower roadmap

New towers must be introduced as small, falsifiable capability probes before
they receive substantial parameter or data budgets. A candidate tower should
not be scaled until it passes all of the following: standalone competence,
lossless typed-request equivalence, held-out dispatch and unsupported-request
rejection, native end-to-end accuracy, causal benefit under ablation, no harm
to unrelated tasks, parallel composition, sequential composition, and measured
conditional-compute savings.

Recommended activation order:

| Order | Candidate tower | Small capability proof before scaling |
| ---: | --- | --- |
| 1 | Code | Sandboxed unit-test pass rate, syntax validity, and typed timeout/failure results. |
| 2 | Tool use | Unseen typed schemas, exact arguments, refusal when no tool applies, and executable mock APIs. |
| 3 | Retrieval/evidence | Supporting-span provenance, multi-document reasoning, citation correctness, and abstention without evidence. |
| 4 | Formal logic | Entailment/contradiction, proof validity, constraints, state tracking, and unsupported abstention. |
| 5 | Structured data | Executable SQL, schema fidelity, table operations, and result equivalence. |
| 6 | Science | Verifiable physics, chemistry, and biology questions with units/evidence separated from unsupported recall. |
| 7 | Information extraction | Exact typed spans, entity classes, overlap handling, relations, and schema-valid output. |
| 8 | Multilingual | Translation, cross-lingual retrieval, language identification, and preservation of names/numbers/formatting. |
| 9 | Long context | Long-range evidence tracking, document comparison, grounded synthesis, and contradiction detection. |
| 10 | Commonsense | Calibrated social/commonsense reasoning with strict no-harm and low false-wake controls. |

The pinned Hugging Face training/evaluation sources, access restrictions, and
license-review boundaries are listed in
[`V2_QWEN_12_TOWER_TARGET.md`](V2_QWEN_12_TOWER_TARGET.md) and encoded in
`config/v2_multi_specialist.yaml`.

Exact string/byte operations should remain a small deterministic specialist;
they are a precision and composition control, not a place where billions of
parameters are likely to buy useful capability. Likewise, retrieval indices,
interpreters, calculators, and API executors should be treated as tools behind
typed contracts when deterministic execution is more reliable than learned
weights.

## Path to a 32B aggregate system

`32B combined` means resident parameters across the coordinator, specialists,
and integration modules. It does not mean that all 32B should be activated for
every request. Relative to the current 4.090B V2 target skeleton, 32B is about
7.8 times larger, so it should be reached by activating and scaling specialists
behind conditional execution rather than replacing the coordinator again.

The exact current non-tower envelope is 4,055,415,136 parameters: the frozen
4,022,468,096-parameter Qwen coordinator plus the dispatcher, Qwen receivers,
and retained integration modules. A 32,000,000,000-parameter resident target
therefore leaves **27,944,584,864 parameters for twelve towers**, averaging
**2,328,715,405 parameters (2.329B) per tower** before allowing for any wider
per-tower adapters.

That average is a budget, not a prescription. Math and code should receive the
largest shares; science, retrieval, long context, logic, and multilingual
should receive medium shares; structured data, extraction, and commonsense can
start smaller; exact byte/string and deterministic tool execution should remain
small. Capacity is increased only after each tower passes its native and routed
gates. Exact sizes must be recomputed from sealed configs because vocabulary,
context embeddings, tied weights, receivers, and modality projections affect
the total.

Scaling gates should be enforced at approximately 0.2B, 1B, 4B, 8B, 16B, and
32B aggregate sizes. At each gate, scale only the components whose standalone
and routed learning curves are still improving; otherwise add missing
capabilities or better data instead of parameters.

## Ordered pipeline

1. prepare and hash broad-math data;
2. train broad math for at least 60 and at most 100 epochs, stopping only after
   10 validation epochs without improvement once the minimum is reached;
3. select the math checkpoint by validation greedy generation;
4. evaluate sealed standalone math generation;
5. record whether later math-data scaling is justified (never auto-scale);
6. prepare and hash multi-specialist/string data;
7. train and seal the learned constrained dispatcher;
8. calibrate frozen Qwen on pure-language prompts;
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

The 26 August 2026 dataset/supervision audit and opt-in bounded repair protocol
are documented in [Math dataset design](docs/MATH_DATASET_DESIGN.md). This is a
diagnostic branch with immutable original data and unchanged production gates;
it does not authorize a long training run or downstream release.

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
support controlled two-specialist cooperation with conditional execution and
a twelve-slot fail-closed dispatcher shape; it would not establish broad
intelligence or competence in any of the ten reserved, untrained towers.

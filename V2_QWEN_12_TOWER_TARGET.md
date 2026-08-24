# V2 dense-Qwen and twelve-tower target

This document is the implementation target for the next V2 run. It is a
training plan, not a claim that the ten reserved experts have already been
trained or accepted.

## Sealed coordinator and dispatcher target

- Coordinator: `Qwen/Qwen3-4B-Instruct-2507` at immutable revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Architecture: dense `Qwen3ForCausalLM`, never an MoE substitute.
- Coordinator shape: 36 decoder layers, hidden width 2,560, BF16 weights, and
  the model's own chat template.
- CFTN receiver layers: 11, 23, and 35 (zero-indexed).
- Dispatcher: 5,025,996 trainable parameters with a frozen 2,560-wide Qwen
  semantic feature, a byte-CNN structural path, and hierarchical heads for
  delegation, multi-tower selection, dependency rounds, and finite typed
  intent graphs.
- Runtime contract: the dispatcher never generates operand values. A typed
  compiler copies immutable source spans, validates dependencies, refuses
  inactive tower slots, and fails closed on unsupported or low-confidence
  prompts.
- Efficiency contract: dispatcher training caches one frozen semantic pass;
  inference reuses the coordinator prepass already required by CFTN. It does
  not load or execute a second Qwen model.

The exact Qwen parameter count reconstructed from the published config is
4,022,468,096. R4 expands the proof math tower from 1,536 to 4,096 lossless
byte tokens and expands the optional answer-composer buffers. The proof math
tower now has exactly 19,023,489 parameters, and all current non-Qwen modules
together add 68,208,738 parameters. The resulting resident skeleton is exactly
**4,090,676,834 parameters (4.091B)**. Only math and string are active today.

For a 32B resident target, reserving the coordinator, dispatcher, and present
non-tower integration leaves approximately 27.945B parameters for twelve
expert towers, or **2.329B per tower on average**. The final allocation should
be heterogeneous: exact byte/string and deterministic tool wrappers should
remain small, leaving more capacity for math, code, science, retrieval, and
long-context experts.

## Math tower: proof run versus scale-up target

The R4 run deliberately keeps the current scratch CFTN math architecture so
the V1.3 mechanism result can be tested without simultaneously replacing the
specialist family. It is not the final high-capability math model.

Current R4 proof tower:

- approximately 19.02M parameters after the larger positional table;
- a lossless 260-token UTF-8 byte vocabulary;
- a 4,096-byte-token context instead of the former 1,536-token ceiling;
- free-form tagged text answers, with the integer categorical head disabled;
- fractions, assignments, systems, symbolic expressions, word problems,
  elementary calculus, probability, polynomial and number-theory data;
- a raw-UTF-8 request and typed-answer result contract. Byte tokenization is an
  internal implementation choice, not part of the CFTN routing protocol;
- open-world non-specialist prompts fall back to the frozen coordinator. They
  are not rejected merely because they lie outside the old archival grammar;
- configured joint task shares are now consumed by the generator exactly:
  15% pure language, 15% explicit math, 15% exact string, 20%
  language-dependent math and 35% multi-specialist, with the final share split
  equally between parallel and sequential examples.

The larger context removes a hard rejection boundary, but it does not by
itself create long-context or graduate-level competence. Context utilization,
standalone generation and native routed generation remain measured gates.

### Later 4B CFTN-native math student

The production scale-up target is an approximately 4B dense math student, not
a 9B teacher embedded as the deployed tower. Strong pretrained models are
training-only teachers. Their weights are absent from the final artifact.

The practical route is:

1. define the final CFTN-native student interface and a shared or explicitly
   registered tower-local tokenizer;
2. initialize from a compatible dense approximately 4B base where licensing
   and architecture permit; a random 4B initialization on one Pod is not a
   credible route to broad language and mathematical competence;
3. continue pretraining on audited mathematical text and notation;
4. perform sequence-level and, where vocabularies align, logit-level
   distillation from stronger dense teachers;
5. retain only teacher outputs that pass deterministic arithmetic, SymPy,
   numerical substitution, unit tests or Lean verification as appropriate;
6. use verifier-guided preference optimization or RL only after supervised
   behavior and answer formatting are stable;
7. freeze the accepted student, train fresh CFTN request/return bridges and
   receivers, and rerun every causal and no-harm control;
8. export the student under `towers.math.*` together with coordinator,
   dispatcher, other towers, bridges, receivers and gates in the unified CFTN
   weights artifact.

Changing from the byte proof tower to a pretrained-initialized student changes
the tower internals and therefore requires fresh evidence. V1.3 still supports
the typed request/result bus, deterministic composition, explicit dispatch,
hard gating and causal-ablation methodology. It does **not** pre-approve new
tokenization, hidden-state bridges, receiver locations, standalone competence,
native routing accuracy or checkpoint serialization.

### Cumulative school-to-research curriculum

Promotion is cumulative: each stage mixes 25-30% replay from earlier stages
and must pass both old and new sealed panels before advancing.

1. arithmetic: signed numbers, fractions, decimals, ratios, units and
   estimation;
2. school algebra and geometry: equations, inequalities, functions,
   coordinate and Euclidean geometry;
3. precalculus and competition foundations: combinatorics, probability,
   number theory and proof-style problems;
4. calculus and differential equations: symbolic and numerical methods with
   domain/constant checks;
5. undergraduate linear algebra, discrete mathematics, statistics and
   numerical analysis;
6. proof-oriented real/complex analysis, abstract algebra and topology;
7. graduate algebra, analysis, geometry, probability, optimization and
   mathematical physics, partitioned by auditable subject panels;
8. formal theorem proving with Lean/mathlib and explicit proof checking;
9. tool-integrated reasoning with typed Python, SymPy and Lean calls;
10. verifier-guided RL on tasks with reliable automatic rewards.

Candidate scale-up sources include
[`nvidia/OpenMathReasoning`](https://huggingface.co/datasets/nvidia/OpenMathReasoning),
[`AI-MO/NuminaMath-1.5`](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5),
OpenR1's Math datasets,
[`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath),
Proof-Pile-2 and
[`mathlib-initiative/mathlib-tactics`](https://huggingface.co/datasets/mathlib-initiative/mathlib-tactics).
Every source requires a pinned revision, provenance and license audit, deduped
held-out problems and contamination checks. "PhD level" means verified
graduate-course and formal-proof performance; it is not a claim of novel
research ability.

With a 32B resident budget, a 4B coordinator, 4B math tower and approximately
0.1-0.5B of shared routing/integration leave roughly 23.5-23.9B for the other
eleven towers, about 2.1B each on average before heterogeneous allocation.

## Tower registry and Hugging Face data

`open` means `datasets.load_dataset` can access the source without accepting
extra terms. Every source must still be pinned by dataset revision, converted
to a hashed local manifest, license-audited for the intended use, and split so
that evaluation examples never enter training.

| Slot | Tower | Training source | Held-out evaluation | Initial proof |
| ---: | --- | --- | --- | --- |
| 1 | Broad math (active) | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) plus the existing deterministic, DeepMind Mathematics, and MathQA mix | Existing V2 math panels and sealed GSM8K/MathQA splits | Exact standalone generation, numeric extrapolation, and routed math benefit |
| 2 | Exact byte/string (active) | Project deterministic generator | Generated disjoint templates, lengths, and compositions | Exact bytes, no normalization drift, parallel and sequential composition |
| 3 | Code | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) train | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) test only | Sandboxed unit-test pass rate, syntax validity, timeout/failure typing |
| 4 | Formal logic | [`tasksource/ruletaker`](https://huggingface.co/datasets/tasksource/ruletaker) train | [`tasksource/folio`](https://huggingface.co/datasets/tasksource/folio), after its broad `cc` license tag is reviewed | Entailment, contradiction, proof-step validity, unsupported abstention |
| 5 | Science | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) ARC-Challenge train | ARC-Challenge validation/test | Answer accuracy with evidence/units separated from unsupported recall |
| 6 | Retrieval/evidence | [`rajpurkar/squad`](https://huggingface.co/datasets/rajpurkar/squad) and [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa) train | HotpotQA validation | Supporting-span provenance, multi-hop accuracy, abstention without evidence |
| 7 | Long context | [`allenai/qasper`](https://huggingface.co/datasets/allenai/qasper) train | [`zai-org/LongBench`](https://huggingface.co/datasets/zai-org/LongBench) as an evaluation benchmark after card/license review | Long-range evidence tracking and citation-grounded synthesis |
| 8 | Multilingual | [`google/wmt24pp`](https://huggingface.co/datasets/google/wmt24pp) train | [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus), gated and evaluation-only | Translation, language ID, name/number/format preservation |
| 9 | Tool use | [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) train | Sealed unseen schemas and executable mock APIs | Function selection, typed arguments, refusal when no tool applies |
| 10 | Structured data | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) train | [`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider) validation after card/license review | Executable SQL, schema fidelity, result equivalence, injection resistance |
| 11 | Information extraction | [`DFKI-SLT/few-nerd`](https://huggingface.co/datasets/DFKI-SLT/few-nerd) supervised train | Few-NERD supervised test | Exact typed spans, entity classes, overlap handling, schema-valid output |
| 12 | Commonsense/social reasoning | [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa) train | CommonsenseQA validation | Choice accuracy, calibrated confidence, no leakage into factual retrieval |

Optional data is not part of the default unattended download. In particular,
`Salesforce/xlam-function-calling-60k` requires accepting Hugging Face access
terms; FLORES+ is protected as an evaluation benchmark; FOLIO, Spider, and
LongBench require an explicit dataset-card/license audit before use. The
default registry records these boundaries instead of silently treating every
Hub repository as unrestricted training data.

## Activation order

Reserved slots are present in the dispatcher's output shape but are masked out
of its loss and impossible to select at runtime. Activate one tower at a time:

1. code, because unit tests provide a strong native competence oracle;
2. tool use, because typed schemas exercise dispatch without open-ended answer
   composition;
3. retrieval/evidence, because provenance makes causal benefit measurable;
4. formal logic and structured data;
5. science and information extraction;
6. multilingual and long context;
7. commonsense last, because its boundary with the general coordinator is the
   least crisp and therefore the easiest place to create unnecessary wakes.

Each activation needs a sealed adapter, native competence test, dispatch
examples including unsupported near-neighbours, no-harm checks, individual
tower disablement, parallel/sequential composition, and measured conditional
compute. Parameter scaling follows evidence: begin with a small probe, then
increase capacity only while standalone and routed validation curves improve.

## RunPod readiness

The startup preflight downloads only the pinned Qwen config/tokenizer, verifies
the full revision, chat-template behavior, dense architecture, hidden width,
layer count, CUDA/BF16 capability, storage, and W&B credentials. Model weights
are downloaded only when a training stage actually needs them.

```bash
cd /workspace/CFTN-MATHS
python run_v2.py --preview
python run_v2.py --preflight-only
```

Do not start the full run until the preview and preflight both pass and the
artifact/data roots point at persistent storage.

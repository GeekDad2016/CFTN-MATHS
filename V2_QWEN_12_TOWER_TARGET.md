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
4,022,468,096. The current two active towers, CFTN integration modules, Qwen
receivers, and dispatcher add 67,057,762 parameters, giving a current V2 target
skeleton of **4,089,525,858 parameters (4.090B)**. Only math and string are
active today.

For a 32B resident target, reserving the coordinator, dispatcher, and present
non-tower integration leaves approximately 27.945B parameters for twelve
expert towers, or **2.329B per tower on average**. The final allocation should
be heterogeneous: exact byte/string and deterministic tool wrappers should
remain small, leaving more capacity for math, code, science, retrieval, and
long-context experts.

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

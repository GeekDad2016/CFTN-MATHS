# CFTN-Text

CFTN-Text is a controlled two-tower experiment: a frozen GPT generalist and a
scratch-trained mathematical specialist exchange compact, dynamically gated
messages in both directions. Both towers execute for every example. There is no
router, top-k expert selection, load-balancing loss, or mixture of final logits.

The first benchmark is exact solution of generated one-variable integer linear
equations, including unseen wording templates and numerical extrapolation.

## Quick start

```powershell
cd G:\ctfn-text
py -3.11 -m pytest
py -3.11 -m tools.prepare_data --config config/v1_linear_equations.yaml
py -3.11 -m tools.prepare_synergy_benchmark --config config/v1_linear_equations.yaml --protocol config/synergy_v1.yaml
py -3.11 -m tools.evaluate_gpt_baseline --config config/v1_linear_equations.yaml
py -3.11 -m tools.train_math_tower --config config/v1_linear_equations.yaml
py -3.11 -m tools.evaluate_math_tower --config config/v1_linear_equations.yaml
py -3.11 -m tools.train_bridges --config config/v1_linear_equations.yaml --stage m2g
py -3.11 -m tools.train_bridges --config config/v1_linear_equations.yaml --stage bidirectional --view-mode complementary
py -3.11 -m tools.evaluate_synergy --config config/v1_linear_equations.yaml --protocol config/synergy_v1.yaml
```

Preview the complete ordered command plan without starting training:

```powershell
py -3.11 -m tools.run_experiment --config config/v1_linear_equations.yaml
```

Add `--execute` to run it. Long-running stages write `status.json` and
append-only `metrics.jsonl` in their artifact directories. See [guide.md](guide.md)
for the architecture contract, controls, and preregistered success criteria.

## Algorithmic V1.1

V1.1 corrects an unfair V1 extrapolation setup. V1 trained a categorical
integer head only on answers from -50 to 50 and then tested classes from 51 to
200. V1.1 disables that auxiliary classifier, judges exact generated
sign-and-digit answers, and uses four progressively expanding numeric bands.
It separates larger-input extrapolation from genuinely unseen-answer
extrapolation so the two failures cannot be conflated.

The exact checkpoints, hashes, completed metrics, causal findings, failed
claims, and unresolved Stage 9 decision for the active seed-719 run are recorded
in [V1_1_RUN_REVIEW.md](V1_1_RUN_REVIEW.md). Read that evidence record before
interpreting individual training or evaluation metrics.

The ordered runner executes all 15 preparation, training, evaluation, causal
control, fixed-open comparison, and evidence stages without manual handoffs:

```powershell
py -3.11 -m tools.run_experiment --config config/v1_1_algorithmic_linear_equations.yaml --synergy-protocol config/synergy_v1_1.yaml --execute --include-fixed-open --wandb --wandb-project cftn-text --wandb-run-name v1-1-algorithmic
```

To wait for an existing stage before starting that fresh pipeline, use the
local continuation process. It polls on the machine and does not consume model
tokens while idle:

```powershell
py -3.11 -m tools.wait_then_run_experiment --wait-status G:\ctfn-text\artifacts\v1_linear_equations\bridge_m2g_contextual\status.json --config config/v1_1_algorithmic_linear_equations.yaml --synergy-protocol config/synergy_v1_1.yaml --poll-seconds 300 --include-fixed-open --wandb --wandb-project cftn-text --wandb-run-name v1-1-algorithmic
```

Pipeline-level progress is written to `pipeline_status.json`; the waiting
handoff is recorded in `continuation_status.json`.

Math-tower training is deliberately blocked until the frozen-GPT calibration
report exists and confirms at least 20 percentage points of capability
headroom. The calibration split is never used to update any model.

The causal collaboration benchmark uses complementary private views. GPT sees
the semantic roles of opaque slots but no numeric values; the math tower sees
the slot values but not their roles. The role permutation changes on every
example. This prevents either tower from solving the proof task alone and makes
disabled, shuffled, directional, and counterfactual bridge interventions
meaningful. It supplements rather than replaces evaluation where both towers
receive the ordinary full prompt. See
[evaluation_protocol.md](evaluation_protocol.md).

Run a disposable one-batch integration test through the math, one-way, and
bidirectional stages on the GPU:

```powershell
py -3.11 -m tools.smoke_test --device cuda
```

## Weights & Biases logging

Install the declared W&B client and provide authentication through the process
environment or the normal `wandb login` credential store. Never place an API
key in this repository or its YAML files.

Enable direct logging for a new stage with `--wandb`:

```powershell
py -3.11 -m tools.train_bridges --stage m2g --wandb --wandb-project cftn-text
```

The ordered experiment runner propagates W&B logging to every training stage,
assigns a distinct run name, and groups the stages together. Its
`--wandb-run-name` value is used as the run-name prefix:

```powershell
py -3.11 -m tools.run_experiment --execute --wandb --wandb-project cftn-text --wandb-run-name synergy-v1
```

For a trainer that was already running before W&B was enabled, backfill and
follow its append-only metrics with the sidecar:

```powershell
py -3.11 -m tools.wandb_watch --run-dir G:\ctfn-text\artifacts\v1_linear_equations\math --wandb --wandb-project cftn-text --poll-seconds 30
```

The sidecar persists only the public W&B run ID, URL, and synchronization
cursor. API keys are never copied into run metadata or model checkpoints.

To resume an interrupted or early-stopped math run while deliberately ignoring
the configured patience limit, use a runtime override. This preserves the
hashed experiment configuration and resumes optimizer and scheduler state:

```powershell
py -3.11 -m tools.train_math_tower --config config/v1_linear_equations.yaml --device cuda --resume --disable-early-stopping --wandb --wandb-project cftn-text
```

## Conditional-communication V1.2

V1.2 is a bridge-only revision between V1.1 and broad-math V2. It reuses the
frozen V1.1 GPT and math towers, preserves the successful math-to-GPT return
path, and retrains only GPT-to-math communication plus its math-side receiver
gates. Every target appears in a complementary view where communication is
required and a shared view where communication must be residual-neutral.

The objective combines task loss, frozen-specialist preservation, a
correct-versus-shuffled message margin, and a soft redundant-view gate penalty.
Best-checkpoint selection uses a fixed greedy-generation causal panel rather
than teacher forcing alone. The preregistered evidence, losses, controls and
acceptance criteria are in
[V1_2_BRIDGE_REVISION.md](V1_2_BRIDGE_REVISION.md).

Preview the isolated five-stage pipeline:

```powershell
py -3.11 -m tools.run_v1_2_experiment --revision-config config/v1_2_conditional_bridge.yaml --device cuda --wandb
```

After the sealed V1.1 pipeline has completed, execute it with:

```powershell
py -3.11 -m tools.run_v1_2_experiment --revision-config config/v1_2_conditional_bridge.yaml --device cuda --wandb --execute
```

The prerequisite audit refuses to launch training while V1.1 is still active
or its reports/checkpoint hashes are incomplete. V1.2 writes only under its own
artifact root and never modifies V1.1 artifacts.

Stage 5 also renders `V1_2_EXPERIMENT_RESULTS.md` from the sealed evidence. The
version-controlled document records what passed, what failed, why the
ablations suggest particular failure hypotheses, and whether to proceed or run
a targeted repair.

## Wake-gated multi-specialist V1.3

If V1.2 passes, V1.3 is the next mechanism experiment. It combines the math
specialist with a new exact-string specialist and tests prompts requiring no
specialist, one specialist, or both specialists. GPT alone receives the raw
prompt; independent wake gates activate the relevant towers, bounded latent
messages carry requests and results, and up to three recurrent callosal rounds
allow sequential expert dependencies.

The complete task matrix, training sequence, ablations, compute measurements,
and fixed acceptance criteria are preregistered in
[V1_3_EXPERIMENT_PLAN.md](V1_3_EXPERIMENT_PLAN.md). Its eventual findings are
recorded in [V1_3_EXPERIMENT_RESULTS.md](V1_3_EXPERIMENT_RESULTS.md).

The isolated V1.3 implementation is now under
`config/v1_3_multi_specialist.yaml`, `cftn_text/v1_3_*`, and the matching
`tools/*v1_3*` entry points. Preview its 12 ordered stages and epoch limits:

```powershell
py -3.11 -m tools.run_v1_3_experiment --config config/v1_3_multi_specialist.yaml --device cuda
```

The automatic continuation waits without consuming GPU compute, validates the
sealed V1.2 report and checkpoint hashes, and launches exactly once only on a
full V1.2 pass:

```powershell
py -3.11 -m tools.wait_then_run_v1_3 --config config/v1_3_multi_specialist.yaml --device cuda --poll-seconds 300 --wandb
```

Every direct V1.3 training/evaluation entry point repeats the same prerequisite
audit, so bypassing the continuation cannot start the experiment early.

### V1.3 LAN inference console

Run the confirmed learned typed-dispatch path from a browser, inspect its full
round-by-round trace, and disable GPT, math, or string independently:

```powershell
D:\Applio-3.2.4\env\python.exe -m tools.serve_v1_3_inference --host 0.0.0.0 --port 7860 --device cuda
```

Then open `http://<computer-ip>:7860/` from this computer or another device on
the same trusted private network. The service has no authentication and must
not be exposed directly to the internet. Artifact defaults, firewall guidance,
API details, trace semantics, and supported task scope are documented in
[V1_3_WEB_INFERENCE.md](V1_3_WEB_INFERENCE.md).

## Broad-math V2

V2 expands the proof to 400,000 unique math examples around a frozen, dense
`Qwen/Qwen3-4B-Instruct-2507` coordinator at a full immutable Hugging Face
revision. It retains the same bridge classes and repeats and scales the full
V1.3 capability sequence rather than inheriting V1.3 as a prerequisite: a
larger math specialist, a larger exact-string specialist, one-round capacity,
dense recurrent communication, supervised soft wakes, a zero-update hard
baseline, gate-only hardening, and real conditional specialist execution.
Earlier V1.2/V1.3 reports are informational provenance only; they neither
initialize nor block V2 collaboration training.
Its curriculum includes variables on both
sides, nested parentheses, signed fractions, two-variable systems, 2–4-step
word problems, distractors, held-out paraphrases, and numerical extrapolation.
The fixed 400K mix is 150,000 project-generated exact problems, 212,690
balanced DeepMind problems, all 29,837 MathQA training programs, and all 7,473
GSM8K training problems. The tower is still trained by autoregressive teacher
forcing over UTF-8 bytes. This revision preserves raw prompts and
source-native programs/traces where available; it does not replace learning
with a symbolic solver. GSM8K-test, MathQA validation/test, and GSM-Symbolic
remain sealed generalization benchmarks.

The math answer head is disabled and checkpoint selection is based first on
validation greedy-generation accuracy. Both native towers are frozen during
integration. V2 trains fresh request/return bridges and receivers, then learns
soft wake and halt gates. Before hardening, the best soft checkpoint is
evaluated in hard mode with zero optimizer updates and with hard halt disabled.
The first hard transition calibrates only the wake gates with required-set BCE,
caps their learning rate at `5e-7`, and refuses to select routing-collapse
checkpoints. Hard halt remains frozen and diagnostic until a separate later
experiment calibrates it.

The public-prompt dispatcher is now a 5,025,996-parameter hierarchical planner:
it combines the frozen Qwen prepass with a byte-CNN structural path, predicts
delegation/towers/dependency rounds/a finite typed graph, and copies operands
from immutable source spans. The registry has twelve slots. Math and string are
active; code, formal logic, science, retrieval, long context, multilingual,
tool use, structured data, information extraction, and commonsense remain
masked `reserved_inactive` slots that consume no tower compute or optimization.
The exact current target skeleton is 4,089,525,858 parameters (4.090B). See
[V2_EVIDENCE_REVISION.md](V2_EVIDENCE_REVISION.md) and the dataset/activation
roadmap in [V2_QWEN_12_TOWER_TARGET.md](V2_QWEN_12_TOWER_TARGET.md).

Preview or execute the complete resumable experiment. The top-level launcher
enables W&B and safe resume by default:

```powershell
py -3.11 run_v2.py --preview
py -3.11 run_v2.py --preflight-only
py -3.11 run_v2.py
```

On a RunPod checkout, installation, persistent-path setup, preflight, and the
resumable launch are combined into one phone-friendly command:

```bash
bash start_v2_runpod.sh
```

The launcher executes 19 resumable stages from broad-math data preparation
through the sealed multi-specialist causal report. It requires neither a V1.3
checkpoint nor a V1.3 report because all V2 collaboration modules are trained
afresh under the V2 revision.

For online V2 logging, the runner requires `WANDB_API_KEY` in the process
environment and never stores its value. `WANDB_PROJECT`, `WANDB_GROUP`, and
`WANDB_ENTITY` are optional. RunPod setup, persistent-volume paths, dataset
boundaries, stages, and logs are documented in
[RUNPOD_V2.md](RUNPOD_V2.md).

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

## Broad-math V2

V2 expands the proof to 400,000 unique examples while keeping GPT-2 frozen and
retaining the same bridge classes. Its curriculum includes variables on both
sides, nested parentheses, signed fractions, two-variable systems, 2–4-step
word problems, distractors, held-out paraphrases, and numerical extrapolation.
DeepMind-generated mathematics and GSM8K-train supply breadth; GSM8K-test and
GSM-Symbolic remain sealed generalization benchmarks.

Preview or execute the complete resumable experiment:

```powershell
py -3.11 -m tools.run_v2_experiment --config config/v2_broad_math.yaml --wandb
py -3.11 -m tools.run_v2_experiment --config config/v2_broad_math.yaml --device cuda --execute --resume --wandb
```

For online V2 logging, the runner requires `WANDB_API_KEY` in the process
environment and never stores its value. RunPod setup, persistent-volume paths,
dataset boundaries, stages, and logs are documented in
[RUNPOD_V2.md](RUNPOD_V2.md).

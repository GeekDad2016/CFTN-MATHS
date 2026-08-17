# CFTN-Text V2 on RunPod

V2 is one resumable end-to-end experiment. It keeps GPT-2 frozen, retains the
successful contextual message bridges and gated cross-receivers, and trains the
scratch math tower on 400,000 unique mixed examples before training the bridges.

## Architecture sequencing after V1.1

The V2 runner now enforces the completed V1.1/V1.2 lessons in code. It does not
launch the old unrestricted bidirectional objective. Conditional GPT-to-math
training freezes the math-to-GPT path and carries forward specialist
preservation, the required-message causal margin, paired required/redundant
views, generation-led checkpoint selection, and a shared-view no-harm gate.

The prerequisite stage verifies a passing V1.2 report, a passing V1.3 report,
their concrete wake/no-harm/causality gates, and the hash chain connecting
them. It runs after the standalone broad-math evaluation but before any bridge
training. The sealed V1.2 report is included in `evidence/`. V1.3 is still
active, so its report is intentionally absent. This does not block independent
math training; it fails closed at the communication boundary. Supply the final
immutable report through `evidence/v1_3_final_report.json` or
`CFTN_V1_3_REPORT`; there is no bypass for a failed report.

The intended division of labour is asymmetric and complementary:

- GPT owns the untouched natural-language problem, discourse context, and final
  response;
- the math specialist owns exact execution in a narrow native workspace and is
  not expected to become a second general-language model;
- GPT-to-math messages provide the operation, semantic roles, constraints, or
  relevant span that the specialist's local input does not contain;
- math-to-GPT messages return the exact result and compact supporting state.

The V2 generated records implement a controlled private-view test:
code prepares a semantic `gpt_problem` and an opaque numeric `math_problem`
before inference. This is appropriate for proving causal bridge cooperation,
but it is not yet autonomous expert use. A later natural-interface arm must
give only GPT the raw prompt, initialize the specialist from neutral workspace
tokens, and test whether GPT forms a sufficient expert request through the
bridge. The final report explicitly does not treat this as a new proof that
V1.3's natural-prompt wake runtime works with the broad specialist. That
checkpoint transfer and matched re-evaluation is the next integration revision
after both V1.3 and this broad specialist pass.

## Data boundary

The 400,000-example training split is fixed before optimization:

| Source | Training examples | Use |
| --- | ---: | --- |
| CFTN deterministic generator | 150,000 | both-side variables, nested parentheses, signed fractions, systems, multi-step and distractor word problems |
| DeepMind Mathematics Dataset generator | 212,690 | balanced algebra, arithmetic, calculus, comparison, measurement, number theory, polynomial, and probability curriculum |
| MathQA official train split | 29,837 | natural word problems with source-native operation programs and answer choices |
| GSM8K official train split | 7,473 | natural multi-step word problems |

The training mechanism is unchanged: the byte-level causal Transformer learns
the next byte of a complete target trace under teacher forcing and is judged by
free greedy generation. V2.1 records preserve `raw_problem`, `native_program`,
`execution_trace`, and the final answer. Project-generated records retain exact
`<work>` traces, GSM8K retains calculator annotations, and MathQA retains its
formal `<program>`. A source that publishes only question/answer pairs remains
truthfully answer-only rather than receiving an invented derivation. Data
preparation computes the exact byte-token length of every sequence and fails
before training if any record exceeds the configured context.

GSM8K test, MathQA validation/test, and all three configured GSM-Symbolic
variants are evaluation-only.
This prevents the proposed generalization benchmark from leaking into training.
GSM-Symbolic files are downloaded at runtime, are not committed to Git, and are
subject to Apple's upstream sample-code license. MathQA is Apache-2.0, GSM8K is
MIT, and the DeepMind generator is Apache-2.0.

The container pins NumPy below 2.0 because the upstream DeepMind 1.0.1
generator still uses an array API removed in NumPy 2.x. A modern-SymPy import
compatibility shim and the generator's former `np.object` alias are applied
locally without downgrading PyTorch's SymPy.

## RunPod launch

### Existing pod / Git checkout (simple launch)

On a new RunPod pod with the repository already cloned or pulled, use a
persistent volume for data, artifacts, model downloads, and W&B files:

```bash
cd /workspace/CFTN-MATHS
git pull --ff-only origin main
python -m pip install --upgrade pip
python -m pip install -e .

export CFTN_DATA_ROOT=/workspace/volume/cftn-text/data/v2_broad_math_400k_r2
export CFTN_ARTIFACT_ROOT=/workspace/volume/cftn-text/artifacts/v2_broad_math_400k_r2
export HF_HOME=/workspace/volume/cftn-text/cache/huggingface
export WANDB_DIR=$CFTN_ARTIFACT_ROOT/wandb
export CFTN_V1_3_REPORT=/workspace/volume/cftn-text/evidence/v1_3_final_report.json

export WANDB_API_KEY='your-runpod-secret'
export WANDB_PROJECT=cftn-text-v2
export WANDB_GROUP=broad-math-400k
# Optional when the API key's default W&B entity is not the desired one:
# export WANDB_ENTITY=your-team-or-user

python run_v2.py --preflight-only
python run_v2.py
```

`python run_v2.py` always requests safe resume and enables online W&B logging.
It validates Python, writable storage, CUDA, BF16, and the W&B key before
starting, then holds an operating-system lock under the artifact root so a
second launcher cannot duplicate the run. The lock is released automatically
if the process or pod dies; running the same command resumes retained stages
and checkpoints. Use `python run_v2.py --no-wandb` only when logging is
intentionally disabled.

If V1.3 has not sealed when standalone math evaluation completes, Stage 5 will
stop before bridge training. Place the immutable report at
`$CFTN_V1_3_REPORT` and run `python run_v2.py` again; Stages 1-4 are validated
and skipped.

### Container plus authenticated monitoring API

1. Build `Dockerfile.runpod` and attach a persistent volume at `/workspace/volume`.
2. Add `WANDB_API_KEY` and a random `CFTN_CONTROL_API_TOKEN` (at least 32
   characters) as RunPod secrets. Do not put either value in the image, YAML,
   command line, or Git.
3. Expose HTTP port `8000` in the Pod template. RunPod's HTTPS proxy URL is
   `https://POD_ID-8000.proxy.runpod.net`; the proxy is public, so bearer
   authentication is mandatory.
4. Optionally set the paths shown in `.env.example`. Set
   `CFTN_CONTROL_ALLOW_UPDATES=1` only if authenticated fast-forward updates are
   required.
5. Start the container without overriding its command.

On first launch, the entrypoint clones `main` into the persistent volume. It
then executes the supervisor, which starts the resumable pipeline and API:

```bash
python -m tools.runpod_supervisor \
  --config config/v2_broad_math.yaml --device cuda \
  --host 0.0.0.0 --port 8000
```

The same command safely resumes after pod interruption. Every stage has a
completion artifact and separate logs under
`${CFTN_ARTIFACT_ROOT}/pipeline_logs`; training metrics, validation curves,
learning rate, throughput, gates, and run summaries are also sent to W&B.

## Authenticated monitoring and safe updates

The API exposes no shell and cannot push to Git. Its mutation surface is
limited to pausing after a completed stage, resuming the existing pipeline, and
fast-forwarding a clean checkout to an exact commit already on `origin/main`.
An update is refused while a stage is active, if the worktree is dirty, if the
remote URL changes, or if the commit is not a published fast-forward.

On a trusted monitoring machine, keep connection values in the environment:

```bash
export CFTN_REMOTE_API_URL="https://POD_ID-8000.proxy.runpod.net"
export CFTN_REMOTE_API_TOKEN="the same RunPod secret value"
python -m tools.runpod_remote status
python -m tools.runpod_remote logs --stage train_math --stream stderr --lines 100
python -m tools.runpod_remote checkpoints
```

For a correction during an active stage, request a safe boundary and wait until
status reports `pipeline.state=paused` and no pipeline process. Commit and push
the correction from the development checkout, then provide both exact commits:

```bash
python -m tools.runpod_remote pause-after-stage
python -m tools.runpod_remote status
python -m tools.runpod_remote update-resume \
  --expected-current-revision OLD_FULL_SHA \
  --revision NEW_FULL_SHA
```

If a stage has already errored, no pause is necessary. The update job writes
`control/update_state.json`, installs the fast-forwarded checkout, and launches
the normal resumable pipeline. Completed stages are validated and skipped;
training resumes only from retained compatible checkpoints.

Preview all stages without downloading data or training:

```bash
python run_v2.py --preview
```

## Ordered stages

1. Generate and hash the immutable manifests.
2. Train the math tower through all three curriculum phases for 12 epochs.
3. Select among retained checkpoints using validation-only greedy generation.
4. Evaluate standalone exact generation and stop on a failed specialist gate.
5. Audit passed, concrete, hash-chained V1.2/V1.3 evidence before bridges.
6. Train math-to-GPT communication on shared complete prompts.
7. Train conditional GPT-to-math while freezing the return path.
8. Evaluate shared-view specialist no-harm and stop on failure.
9. Run complementary closed, directional, shuffled, and fixed-open causal arms.
10. Assess a later one-million-example run from generated held-out trends;
    scaling is never started automatically.
11. Assemble `v2_final_report.json` and require every preceding gate.

The conservative batches fit smaller development GPUs; a B200 has substantial
headroom and should first run the same hashed baseline. Any throughput-tuned
batch-size profile changes optimization and therefore receives a new config
hash rather than silently resuming this experiment.

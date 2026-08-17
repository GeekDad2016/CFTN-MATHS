# CFTN-Text V2 on RunPod

V2 is one resumable end-to-end experiment. It keeps GPT-2 frozen, retains the
contextual message bridges and gated cross-receivers, trains the scratch math
tower on 400,000 unique mixed examples, trains a larger exact-string tower,
and then repeats the complete recurrent/wake-gated V1.3 curriculum with fresh
collaboration modules.

## Architecture sequencing after V1.1

The V2 runner does not launch the old unrestricted bidirectional objective and
does not require a sealed V1.3 report. V2 trains every collaboration component
from scratch, so V1.2/V1.3 reports are informational provenance only. Both
native towers remain frozen during integration. The curriculum establishes
single-specialist message capacity, dense mixed communication, three-round
recurrence, supervised soft wakes, and finally hard conditional execution.

The soft-to-hard transition incorporates the V1.3 collapse diagnosis directly:
the selected soft checkpoint is first evaluated in hard mode without any
updates; hardening then freezes every bridge and receiver, trains only wake and
halt gates, uses no LR warmup, and caps the gate LR at `5e-7`. Checkpoints with
false-wake, exact-routing, always-open, always-closed, or baseline-regression
failures cannot be selected.

The intended division of labour is asymmetric and complementary:

- GPT owns the untouched natural-language problem, discourse context, and final
  response;
- the math specialist owns exact execution in a narrow native workspace and is
  not expected to become a second general-language model;
- GPT-to-math messages provide the operation, semantic roles, constraints, or
  relevant span that the specialist's local input does not contain;
- math-to-GPT messages return the exact result and compact supporting state.

During joint training GPT receives the natural prompt while each specialist is
initialized from neutral workspace tokens. Pure language requires no tower;
math or exact string tasks require one; language-dependent and composed tasks
require directional or recurrent cooperation. A third `extension_1` slot is
reserved but inactive until its dataset and capability contract are selected;
it consumes no parameters or compute in this run.

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

On a new RunPod pod with the repository already cloned, the phone-friendly
launch is one command:

```bash
cd /workspace/CFTN-MATHS
bash start_v2_runpod.sh
```

The bootstrap fast-forwards a clean `main` checkout, reopens its updated copy,
sets all persistent `/workspace/volume/cftn-text` paths, installs the project
and dependencies, runs the CUDA/BF16/storage/W&B preflight, and starts the
normal resumable launcher. Prefer adding `WANDB_API_KEY` as a RunPod secret. If
it is absent and the terminal is interactive, the bootstrap securely prompts
for it without echoing or writing it to disk.

The pipeline holds an operating-system lock under the artifact root so a
second launcher cannot duplicate the run. The lock is released automatically
if the process or pod dies; running `bash start_v2_runpod.sh` again validates
completed artifacts and resumes retained checkpoints.

Advanced path, W&B project/group/entity, Python executable, branch, and storage
overrides remain available through the variables in `.env.example`. Set
`CFTN_SKIP_GIT_UPDATE=1` only for an intentionally pinned checkout.

No V1.2 or V1.3 report variable is required. Existing reports in `evidence/`
are recorded when present but never gate this fresh V2 run.

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

1. Generate and hash the broad-math manifests.
2. Train the math tower through all three curriculum phases for up to 100
   epochs. The run cannot stop before epoch 60 and thereafter stops only after
   10 validation epochs without improvement; the former 12-epoch fixed cap was
   removed because the specialist was still improving when it ended.
3. Select among retained checkpoints using validation-only greedy generation.
4. Evaluate standalone exact generation and stop on a failed specialist gate.
5. Assess whether later math-data scaling is justified; never auto-scale.
6. Generate and hash exact-string and joint multi-specialist manifests.
7. Calibrate frozen GPT on pure-language prompts.
8. Train the larger exact-string specialist for at most 30 epochs.
9. Seal familiar and task-matched native specialist competence.
10. Train one-round single-specialist capacity for 8 epochs.
11. Train dense mixed communication for 12 epochs.
12. Train dense three-round recurrence for 12 epochs.
13. Train supervised soft wake/halt behavior for 10 epochs.
14. Evaluate the selected soft checkpoint in hard mode with zero updates.
15. Harden only wake/halt gates for at most 10 epochs at no more than `5e-7`.
16. Run closed, directional, shuffled, fixed-open, one-round, serial,
    recurrence, no-harm, routing, synergy, and compute controls.
17. Assemble `v2_final_report.json` and `V2_EXPERIMENT_RESULTS.md`.

The registered batch sizes target a high-memory RTX PRO 6000-class RunPod.
Changing batch size changes optimization and therefore requires a new config
hash rather than silently resuming this experiment.

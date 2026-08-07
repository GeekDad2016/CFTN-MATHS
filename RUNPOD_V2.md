# CFTN-Text V2 on RunPod

V2 is one resumable end-to-end experiment. It keeps GPT-2 frozen, retains the
successful contextual message bridges and gated cross-receivers, and trains the
scratch math tower on 400,000 unique mixed examples before training the bridges.

## Data boundary

The 400,000-example training split is fixed before optimization:

| Source | Training examples | Use |
| --- | ---: | --- |
| CFTN deterministic generator | 100,000 | both-side variables, nested parentheses, signed fractions, systems, 2–4-step and distractor word problems |
| DeepMind Mathematics Dataset generator | 292,527 | easy/medium/hard algebra, arithmetic, and polynomial curriculum |
| GSM8K official train split | 7,473 | natural multi-step word problems |

GSM8K test and all three configured GSM-Symbolic variants are evaluation-only.
This prevents the proposed generalization benchmark from leaking into training.
GSM-Symbolic files are downloaded at runtime, are not committed to Git, and are
subject to their upstream CC BY-NC-ND 4.0 terms.

The container pins NumPy below 2.0 because the upstream DeepMind 1.0.1
generator still uses an array API removed in NumPy 2.x. A modern-SymPy import
compatibility shim and the generator's former `np.object` alias are applied
locally without downgrading PyTorch's SymPy.

## RunPod launch

1. Build `Dockerfile.runpod` and attach a persistent volume at `/workspace/volume`.
2. Add `WANDB_API_KEY` as a RunPod secret. Do not put it in the image, YAML, or
   command line.
3. Optionally set the paths shown in `.env.example`.
4. Start the container without overriding its command.

The entrypoint executes:

```bash
python -m tools.run_v2_experiment \
  --config config/v2_broad_math.yaml \
  --device cuda --execute --resume --wandb
```

The same command safely resumes after pod interruption. Every stage has a
completion artifact and separate logs under
`${CFTN_ARTIFACT_ROOT}/pipeline_logs`; training metrics, validation curves,
learning rate, throughput, gates, and run summaries are also sent to W&B.

Preview all stages without downloading data or training:

```bash
python -m tools.run_v2_experiment --config config/v2_broad_math.yaml --wandb
```

## Ordered stages

1. Generate and hash the immutable manifests.
2. Train the math tower through three difficulty phases for at most 12 epochs.
3. Evaluate standalone exact generation on local, DeepMind, GSM8K, and
   GSM-Symbolic held-out data.
4. Train math-to-GPT communication on complementary private views.
5. Initialize from that checkpoint and train bidirectional communication.
6. Run closed-bridge, directional, shuffled-message, and fixed-open causal arms.
7. Decide whether a later one-million-example run is justified by the recent
   held-out learning curve. Scaling is never started automatically.
8. Assemble `v2_final_report.json`.

The default batch sizes target a 24 GB RunPod GPU. If the selected GPU has less
memory, lower both training batch sizes in the YAML; this changes the experiment
hash and therefore creates a distinct run rather than silently resuming an
incompatible checkpoint.

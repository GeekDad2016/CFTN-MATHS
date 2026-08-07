#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

: "${WANDB_API_KEY:?WANDB_API_KEY must be supplied as a RunPod secret}"

export CFTN_DATA_ROOT="${CFTN_DATA_ROOT:-/workspace/volume/cftn-text/data/v2_broad_math_400k}"
export CFTN_ARTIFACT_ROOT="${CFTN_ARTIFACT_ROOT:-/workspace/volume/cftn-text/artifacts/v2_broad_math_400k}"
export HF_HOME="${HF_HOME:-/workspace/volume/cftn-text/cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-cftn-text-v2}"
export WANDB_DIR="${WANDB_DIR:-${CFTN_ARTIFACT_ROOT}/wandb}"

mkdir -p "${CFTN_DATA_ROOT}" "${CFTN_ARTIFACT_ROOT}" "${HF_HOME}" "${WANDB_DIR}"

exec python -m tools.run_v2_experiment \
  --config config/v2_broad_math.yaml \
  --device cuda \
  --execute \
  --resume \
  --wandb

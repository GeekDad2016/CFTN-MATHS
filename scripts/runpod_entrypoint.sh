#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

: "${WANDB_API_KEY:?WANDB_API_KEY must be supplied as a RunPod secret}"
: "${CFTN_CONTROL_API_TOKEN:?CFTN_CONTROL_API_TOKEN must be supplied as a RunPod secret}"

storage_root="${CFTN_STORAGE_ROOT:-/workspace/cftn-text}"
export CFTN_REPOSITORY_ROOT="${CFTN_REPOSITORY_ROOT:-/workspace/CFTN-MATHS}"
export CFTN_GIT_REPOSITORY="${CFTN_GIT_REPOSITORY:-https://github.com/GeekDad2016/CFTN-MATHS.git}"
export CFTN_GIT_BRANCH="${CFTN_GIT_BRANCH:-main}"
export CFTN_DATA_ROOT="${CFTN_DATA_ROOT:-${storage_root}/data/v2_broad_math_400k_r3}"
export CFTN_V2_MULTI_DATA_ROOT="${CFTN_V2_MULTI_DATA_ROOT:-${storage_root}/data/v2_multi_specialist_r2}"
export CFTN_ARTIFACT_ROOT="${CFTN_ARTIFACT_ROOT:-${storage_root}/artifacts/v2_broad_math_400k_r3}"
export HF_HOME="${HF_HOME:-${storage_root}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${storage_root}/cache/pip}"
export WANDB_PROJECT="${WANDB_PROJECT:-cftn-text-v2}"
export WANDB_GROUP="${WANDB_GROUP:-scaled-multi-specialist-r3}"
export WANDB_DIR="${WANDB_DIR:-${CFTN_ARTIFACT_ROOT}/wandb}"
export CFTN_CONTROL_HOST="${CFTN_CONTROL_HOST:-0.0.0.0}"
export CFTN_CONTROL_PORT="${CFTN_CONTROL_PORT:-8000}"
export CFTN_CONTROL_ALLOW_UPDATES="${CFTN_CONTROL_ALLOW_UPDATES:-0}"

mkdir -p "$(dirname "${CFTN_REPOSITORY_ROOT}")" "${storage_root}" \
  "${CFTN_DATA_ROOT}" "${CFTN_V2_MULTI_DATA_ROOT}" \
  "${CFTN_ARTIFACT_ROOT}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${WANDB_DIR}"

verify_durable_mount() {
  local label="$1"
  local path="$2"
  local mount_target
  if ! command -v findmnt >/dev/null 2>&1; then
    echo "findmnt is required to verify persistent RunPod storage." >&2
    exit 1
  fi
  mount_target="$(findmnt -n -o TARGET -T "${path}")"
  if [[ "${mount_target}" == "/" \
      && "${CFTN_ALLOW_EPHEMERAL_STORAGE:-0}" != "1" ]]; then
    echo "Refusing ephemeral ${label} path ${path}; use /workspace storage." >&2
    exit 1
  fi
}

verify_durable_mount "repository" "$(dirname "${CFTN_REPOSITORY_ROOT}")"
verify_durable_mount "storage" "${storage_root}"

if [[ ! -d "${CFTN_REPOSITORY_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${CFTN_REPOSITORY_ROOT}")"
  git clone --single-branch --branch "${CFTN_GIT_BRANCH}" \
    "${CFTN_GIT_REPOSITORY}" "${CFTN_REPOSITORY_ROOT}"
fi

git config --global --add safe.directory "${CFTN_REPOSITORY_ROOT}"
cd "${CFTN_REPOSITORY_ROOT}"
if [[ "$(git remote get-url origin)" != "${CFTN_GIT_REPOSITORY}" ]]; then
  echo "Refusing to run: persistent checkout origin does not match CFTN_GIT_REPOSITORY" >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != "${CFTN_GIT_BRANCH}" ]]; then
  echo "Refusing to run: persistent checkout is not on ${CFTN_GIT_BRANCH}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run: persistent checkout has uncommitted files" >&2
  exit 1
fi
python -m pip install -e .

exec python -m tools.runpod_supervisor \
  --config config/v2_broad_math.yaml \
  --device cuda \
  --host "${CFTN_CONTROL_HOST}" \
  --port "${CFTN_CONTROL_PORT}"

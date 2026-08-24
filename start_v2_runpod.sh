#!/usr/bin/env bash
set -Eeuo pipefail

# One-file RunPod bootstrap for the complete resumable CFTN-Text V2 run.
# Run from an existing checkout with: bash start_v2_runpod.sh

umask 077

on_error() {
  local exit_code=$?
  echo "V2 bootstrap failed near line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  echo "Fix the reported issue, then run this same file again; completed work resumes safely." >&2
  exit "${exit_code}"
}
trap on_error ERR

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if [[ ! -f pyproject.toml || ! -f run_v2.py ]]; then
  echo "Run this file from the CFTN-MATHS repository checkout." >&2
  exit 1
fi

python_bin="${CFTN_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
  else
    echo "Python 3.11 or newer is required but was not found." >&2
    exit 1
  fi
fi

# Pull once, then reopen the potentially updated bootstrap file. Disable only
# for an intentionally pinned checkout with CFTN_SKIP_GIT_UPDATE=1.
if [[ -d .git && "${CFTN_SKIP_GIT_UPDATE:-0}" != "1" \
    && "${CFTN_BOOTSTRAP_REEXEC:-0}" != "1" ]]; then
  git config --global --add safe.directory "${script_dir}"
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to update a checkout with uncommitted files: ${script_dir}" >&2
    exit 1
  fi
  expected_branch="${CFTN_GIT_BRANCH:-main}"
  current_branch="$(git branch --show-current)"
  if [[ "${current_branch}" != "${expected_branch}" ]]; then
    echo "Expected Git branch ${expected_branch}, found ${current_branch}." >&2
    exit 1
  fi
  echo "Updating ${expected_branch} from origin..."
  git fetch origin "${expected_branch}"
  git merge --ff-only "origin/${expected_branch}"
  export CFTN_BOOTSTRAP_REEXEC=1
  exec bash "${script_dir}/start_v2_runpod.sh" "$@"
fi

storage_root="${CFTN_STORAGE_ROOT:-/workspace/cftn-text}"
export CFTN_DATA_ROOT="${CFTN_DATA_ROOT:-${storage_root}/data/v2_broad_math_400k_r4}"
export CFTN_V2_MULTI_DATA_ROOT="${CFTN_V2_MULTI_DATA_ROOT:-${storage_root}/data/v2_multi_specialist_r3}"
export CFTN_ARTIFACT_ROOT="${CFTN_ARTIFACT_ROOT:-${storage_root}/artifacts/v2_broad_math_400k_r4}"
export HF_HOME="${HF_HOME:-${storage_root}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${storage_root}/cache/pip}"
export WANDB_DIR="${WANDB_DIR:-${CFTN_ARTIFACT_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-cftn-text-v2}"
export WANDB_GROUP="${WANDB_GROUP:-scaled-multi-specialist-r4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "${storage_root}" \
  "${CFTN_DATA_ROOT}" \
  "${CFTN_V2_MULTI_DATA_ROOT}" \
  "${CFTN_ARTIFACT_ROOT}" \
  "${HF_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${WANDB_DIR}"

verify_durable_mount() {
  local label="$1"
  local path="$2"
  local mount_target
  local mount_type

  if ! command -v findmnt >/dev/null 2>&1; then
    if [[ "${CFTN_ALLOW_EPHEMERAL_STORAGE:-0}" == "1" ]]; then
      echo "WARNING: findmnt is unavailable; durability check skipped for ${label}." >&2
      return
    fi
    echo "findmnt is required to verify persistent RunPod storage for ${label}." >&2
    exit 1
  fi

  mount_target="$(findmnt -n -o TARGET -T "${path}")"
  mount_type="$(findmnt -n -o FSTYPE -T "${path}")"
  if [[ "${mount_target}" == "/" \
      && "${CFTN_ALLOW_EPHEMERAL_STORAGE:-0}" != "1" ]]; then
    echo "Refusing ephemeral ${label} path ${path}; it resolves to the container root filesystem." >&2
    echo "Use a /workspace path, or set CFTN_ALLOW_EPHEMERAL_STORAGE=1 only for a disposable smoke test." >&2
    exit 1
  fi
  echo "  ${label} mount: ${mount_target} (${mount_type})"
}

echo "Verifying durable RunPod paths..."
verify_durable_mount "repository" "${script_dir}"
verify_durable_mount "storage" "${storage_root}"

# Keep only the reproducible virtual environment on the Pod's fast local disk.
# Source, data, caches, checkpoints, and reports are durable under /workspace;
# a replacement container can recreate this environment from pyproject.toml.
venv_root="${CFTN_VENV_ROOT:-/opt/cftn-v2-venv}"
if [[ -e "${venv_root}" && ! -x "${venv_root}/bin/python" ]]; then
  echo "CFTN_VENV_ROOT exists but is not a usable virtual environment: ${venv_root}" >&2
  exit 1
fi
if [[ ! -x "${venv_root}/bin/python" ]]; then
  echo "Creating local-disk virtual environment at ${venv_root}..."
  "${python_bin}" -m venv --system-site-packages "${venv_root}"
fi
python_bin="${venv_root}/bin/python"

wandb_required=1
preflight_only=0
for argument in "$@"; do
  if [[ "${argument}" == "--no-wandb" ]]; then
    wandb_required=0
  elif [[ "${argument}" == "--preflight-only" ]]; then
    preflight_only=1
  fi
done

if [[ "${wandb_required}" == "1" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ -t 0 ]]; then
    echo "WANDB_API_KEY was not supplied as a RunPod environment variable."
    read -r -s -p "Paste the W&B API key (input is hidden): " WANDB_API_KEY
    echo
    if [[ -z "${WANDB_API_KEY}" ]]; then
      echo "A non-empty W&B API key is required for the registered V2 run." >&2
      exit 1
    fi
    export WANDB_API_KEY
  else
    echo "Set WANDB_API_KEY as a RunPod secret/environment variable and rerun." >&2
    exit 1
  fi
fi

# Credentials copied through notebooks, PowerShell pipes, or secret editors can
# carry a trailing CR/LF even when the visible key looks correct. W&B rejects
# leading or trailing whitespace, so normalize it once at the process boundary.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(printf '%s' "${WANDB_API_KEY}" | tr -d '[:space:]')"
  if [[ -z "${WANDB_API_KEY}" && "${wandb_required}" == "1" ]]; then
    echo "WANDB_API_KEY contained only whitespace." >&2
    exit 1
  fi
  export WANDB_API_KEY
fi

echo "Installing CFTN-Text and all declared dependencies..."
pip_install_args=()
externally_managed="$("${python_bin}" -c \
  'from pathlib import Path; import sysconfig; print(int((Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file()))')"
if [[ "${externally_managed}" == "1" ]]; then
  echo "Python is externally managed; enabling pip's container-safe override."
  pip_install_args+=(--break-system-packages)
fi
"${python_bin}" -m pip install "${pip_install_args[@]}" --upgrade pip setuptools wheel
"${python_bin}" -m pip install "${pip_install_args[@]}" -e "${script_dir}"

revision="not-a-git-checkout"
if [[ -d .git ]]; then
  revision="$(git rev-parse HEAD)"
fi

echo
echo "CFTN-Text V2 bootstrap"
echo "  revision: ${revision}"
echo "  Python: ${python_bin}"
echo "  virtual environment: ${venv_root}"
echo "  persistent storage: ${storage_root}"
echo "  math data: ${CFTN_DATA_ROOT}"
echo "  multi-specialist data: ${CFTN_V2_MULTI_DATA_ROOT}"
echo "  artifacts: ${CFTN_ARTIFACT_ROOT}"
echo "  Hugging Face cache: ${HF_HOME}"
echo "  pip cache: ${PIP_CACHE_DIR}"
echo "  W&B project/group: ${WANDB_PROJECT}/${WANDB_GROUP}"
echo
echo "Running CUDA, BF16, storage, configuration, and W&B preflight..."
"${python_bin}" run_v2.py --preflight-only "$@"

if [[ "${preflight_only}" == "1" ]]; then
  echo "Preflight-only mode complete; training was not launched."
  exit 0
fi

echo
echo "Preflight passed. Starting or safely resuming the 19-stage V2 pipeline..."
exec "${python_bin}" run_v2.py "$@"

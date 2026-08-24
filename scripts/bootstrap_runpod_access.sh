#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent, one-command RunPod bootstrap. With no arguments it restores SSH
# access, fast-forwards the durable checkout, rebuilds the disposable venv when
# needed, and runs a no-training preflight. Training requires --launch.

umask 077

mode="prepare"
launcher_args=()
case "${1:-}" in
  "")
    ;;
  --access-only)
    mode="access_only"
    shift
    ;;
  --launch)
    mode="launch"
    shift
    launcher_args=("$@")
    ;;
  --help|-h)
    cat <<'EOF'
Usage:
  bash /workspace/cftn-start.sh                 # SSH + Git + dependencies + preflight
  bash /workspace/cftn-start.sh --access-only   # SSH + Git only
  bash /workspace/cftn-start.sh --launch [args] # explicitly launch/resume V2

The default mode never starts training.
EOF
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    echo "Use --help for supported modes." >&2
    exit 2
    ;;
esac

storage_root="${CFTN_STORAGE_ROOT:-/workspace/cftn-text}"
repository_root="${CFTN_REPOSITORY_ROOT:-/workspace/CFTN-MATHS}"
repository_url="${CFTN_GIT_REPOSITORY:-https://github.com/GeekDad2016/CFTN-MATHS.git}"
repository_branch="${CFTN_GIT_BRANCH:-main}"
public_key_file="${CFTN_SSH_PUBLIC_KEY_FILE:-${storage_root}/ssh/id_ed25519_runpod_cftn.pub}"
ssh_home="${CFTN_SSH_HOME:-/root/.ssh}"
authorized_keys="${ssh_home}/authorized_keys"
bootstrap_path="${CFTN_PERSISTENT_BOOTSTRAP:-/workspace/cftn-start.sh}"
connection_note="${storage_root}/ssh/README.txt"

verify_durable_path() {
  local label="$1"
  local path="$2"
  local mount_target
  if ! command -v findmnt >/dev/null 2>&1; then
    echo "findmnt is required to verify persistent RunPod storage." >&2
    exit 1
  fi
  mount_target="$(findmnt -n -o TARGET -T "${path}")"
  if [[ "${mount_target}" == "/" ]]; then
    echo "Refusing ${label} path ${path}: it is on ephemeral container storage." >&2
    exit 1
  fi
}

install_authorized_key() {
  local key_line
  local key_blob
  if [[ ! -s "${public_key_file}" ]]; then
    echo "Persistent SSH public key is missing: ${public_key_file}" >&2
    exit 1
  fi
  ssh-keygen -lf "${public_key_file}" >/dev/null
  key_line="$(tr -d '\r\n' < "${public_key_file}")"
  key_blob="$(printf '%s\n' "${key_line}" | awk '{print $2}')"
  if [[ -z "${key_blob}" ]]; then
    echo "Persistent SSH public key is malformed: ${public_key_file}" >&2
    exit 1
  fi

  install -d -m 700 "${ssh_home}"
  touch "${authorized_keys}"
  chmod 600 "${authorized_keys}"
  if ! grep -Fq -- "${key_blob}" "${authorized_keys}"; then
    printf '%s\n' "${key_line}" >> "${authorized_keys}"
    echo "Installed the persistent SSH public key."
  else
    echo "Persistent SSH public key is already authorized."
  fi
}

start_sshd() {
  if [[ "${CFTN_SKIP_SSHD_START:-0}" == "1" ]]; then
    return
  fi
  if pgrep -x sshd >/dev/null 2>&1; then
    echo "sshd is already running."
    return
  fi
  if ! command -v sshd >/dev/null 2>&1; then
    echo "sshd is not installed in this Pod image." >&2
    exit 1
  fi
  ssh-keygen -A
  install -d -m 755 /run/sshd
  "$(command -v sshd)"
  if ! pgrep -x sshd >/dev/null 2>&1; then
    echo "sshd failed to start." >&2
    exit 1
  fi
  echo "sshd started."
}

update_repository() {
  local before_revision=""
  local after_revision

  if [[ ! -d "${repository_root}/.git" ]]; then
    if [[ -e "${repository_root}" ]]; then
      echo "Repository path exists but is not a Git checkout: ${repository_root}" >&2
      exit 1
    fi
    git clone --single-branch --branch "${repository_branch}" \
      "${repository_url}" "${repository_root}"
  fi

  git config --global --add safe.directory "${repository_root}"
  if [[ "$(git -C "${repository_root}" remote get-url origin)" != "${repository_url}" ]]; then
    echo "Persistent checkout origin does not match CFTN_GIT_REPOSITORY." >&2
    exit 1
  fi
  if [[ "$(git -C "${repository_root}" branch --show-current)" != "${repository_branch}" ]]; then
    echo "Persistent checkout is not on ${repository_branch}." >&2
    exit 1
  fi
  if [[ -n "$(git -C "${repository_root}" status --porcelain)" ]]; then
    echo "Persistent checkout has uncommitted files; refusing to update it." >&2
    exit 1
  fi

  before_revision="$(git -C "${repository_root}" rev-parse HEAD)"
  if [[ "${CFTN_SKIP_REPOSITORY_UPDATE:-0}" != "1" ]]; then
    git -C "${repository_root}" fetch origin "${repository_branch}"
    git -C "${repository_root}" merge --ff-only "origin/${repository_branch}"
  fi
  after_revision="$(git -C "${repository_root}" rev-parse HEAD)"
  echo "Persistent checkout: ${after_revision}"

  install -m 700 \
    "${repository_root}/scripts/bootstrap_runpod_access.sh" \
    "${bootstrap_path}"

  if [[ "${before_revision}" != "${after_revision}" \
      && "${CFTN_BOOTSTRAP_REEXEC:-0}" != "1" ]]; then
    export CFTN_BOOTSTRAP_REEXEC=1
    case "${mode}" in
      prepare) exec "${bootstrap_path}" ;;
      access_only) exec "${bootstrap_path}" --access-only ;;
      launch) exec "${bootstrap_path}" --launch "${launcher_args[@]}" ;;
    esac
  fi
}

write_connection_note() {
  install -d -m 700 "$(dirname "${connection_note}")"
  {
    echo "Run this once from the RunPod web terminal after a Pod opens:"
    echo "  bash /workspace/cftn-start.sh"
    echo
    echo "Then connect from local PowerShell using the current IP and SSH port shown by RunPod:"
    echo '  ssh -p <SSH_PORT> -i "C:\Users\adria\.ssh\id_ed25519_runpod_cftn" root@<PUBLIC_IP>'
  } > "${connection_note}"
  chmod 600 "${connection_note}"
}

mkdir -p "${storage_root}" "$(dirname "${repository_root}")"
verify_durable_path "storage" "${storage_root}"
verify_durable_path "repository" "$(dirname "${repository_root}")"
install_authorized_key
start_sshd
update_repository
write_connection_note

if [[ "${mode}" == "access_only" ]]; then
  echo "CFTN access is ready. No dependency installation or training was run."
  exit 0
fi

if [[ "${mode}" == "launch" ]]; then
  echo "Explicit launch requested; handing off to start_v2_runpod.sh."
  exec bash "${repository_root}/start_v2_runpod.sh" "${launcher_args[@]}"
fi

echo "Preparing the disposable environment and running a no-training preflight..."
bash "${repository_root}/start_v2_runpod.sh" --preflight-only --no-wandb
echo "CFTN Pod is ready. SSH, Git, dependencies, GPU, and persistent paths passed."

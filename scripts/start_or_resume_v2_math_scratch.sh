#!/usr/bin/env bash
set -euo pipefail

cd /workspace/CFTN-MATHS
if [[ -x /opt/cftn-v2-venv/bin/python ]]; then
  python_bin=/opt/cftn-v2-venv/bin/python
elif [[ -x /opt/cftn-data-pilot-venv/bin/python ]]; then
  python_bin=/opt/cftn-data-pilot-venv/bin/python
else
  echo "No verified CFTN RunPod Python environment is installed." >&2
  exit 1
fi
exec "$python_bin" -u -m tools.run_v2_math_curriculum auto

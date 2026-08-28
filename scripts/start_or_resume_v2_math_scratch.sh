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
args=(auto)
if [[ -d /workspace/cftn-text/artifacts/v2_broad_math_400k_r4/math_scratch_curriculum_v4 ]]; then
  # The active V4 artifact has an immutable runtime record for its cheaper,
  # phase-required generation validation schedule.  A new artifact remains on
  # the sealed default until an explicit future policy change is recorded.
  args+=(--generation-panel-scope phase_required_v1)
fi
exec "$python_bin" -u -m tools.run_v2_math_curriculum "${args[@]}"

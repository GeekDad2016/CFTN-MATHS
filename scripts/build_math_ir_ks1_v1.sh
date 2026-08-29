#!/usr/bin/env bash
set -euo pipefail

project_root="${CFTN_PROJECT_ROOT:-/workspace/CFTN-MATHS}"
output_root="${CFTN_MATH_IR_DATA_ROOT:-/workspace/cftn-text/data/math_ir_ks1_v1}"
python_bin="${CFTN_PYTHON:-/opt/cftn-v2-venv/bin/python}"

cd "$project_root"
"$python_bin" -m tools.build_math_curriculum_dataset prepare \
  --config config/math_ir_ks1_curriculum_v1.json \
  --output "$output_root"
"$python_bin" -m tools.build_math_curriculum_dataset summary \
  --output "$output_root"

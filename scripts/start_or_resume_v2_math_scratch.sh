#!/usr/bin/env bash
set -euo pipefail

cd /workspace/CFTN-MATHS
exec /opt/cftn-v2-venv/bin/python -u -m tools.run_v2_math_curriculum auto

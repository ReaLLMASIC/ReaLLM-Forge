#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"
LAB_INDEX=https://download.pytorch.org/whl/cu128
if [[ "${1:-}" == --cpu ]]; then
  LAB_INDEX=https://download.pytorch.org/whl/cpu
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash scripts/setup.sh [--cpu]' >&2
  exit 2
fi
"${LAB_BOOTSTRAP_PYTHON:-python3}" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), "Use Python 3.10–3.13"'
"${LAB_BOOTSTRAP_PYTHON:-python3}" -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch==2.9.1 --index-url "$LAB_INDEX"
venv/bin/python -m pip install triton==3.5.1 -r requirements.txt third_party/wheels/*.whl
venv/bin/python scripts/check_install.py
echo 'Ready. Inspect: bash run.sh plan. Offline check: bash run.sh smoke. Full GPU sweep: bash run.sh all.'

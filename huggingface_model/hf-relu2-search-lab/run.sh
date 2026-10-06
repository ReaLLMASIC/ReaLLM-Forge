#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_ROOT"
LAB_EXECUTABLE="${LAB_PYTHON:-$LAB_ROOT/venv/bin/python}"
if [[ ! -x "$LAB_EXECUTABLE" ]]; then
  echo 'First run: bash scripts/setup.sh (or --cpu for offline smoke only).' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
exec "$LAB_EXECUTABLE" -m context_sweep.launch "$@"

#!/usr/bin/env bash
set -euo pipefail
DEPTH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEPTH_ROOT"
DEPTH_PYTHON="${LAB_PYTHON:-$DEPTH_ROOT/venv/bin/python}"
if [[ ! -x "$DEPTH_PYTHON" ]]; then
  echo 'First run: bash scripts/setup.sh (or set LAB_PYTHON to the prior lab venv/bin/python).' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
exec "$DEPTH_PYTHON" -m depth_sweep.launch "$@"

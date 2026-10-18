#!/usr/bin/env bash
set -euo pipefail
QUICK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$QUICK_ROOT"
QUICK_PYTHON="${LAB_PYTHON:-$QUICK_ROOT/venv/bin/python}"
if [[ ! -x "$QUICK_PYTHON" ]]; then
  echo 'Run bash scripts/setup.sh, or set LAB_PYTHON to the existing lab venv/bin/python.' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
exec "$QUICK_PYTHON" -m quick_depth.launch "$@"

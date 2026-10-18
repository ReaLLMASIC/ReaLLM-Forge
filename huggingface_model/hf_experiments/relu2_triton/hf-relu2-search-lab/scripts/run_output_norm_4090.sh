#!/usr/bin/env bash
# Compatibility entry point for the preserved KDA/output-norm preset.
set -euo pipefail
LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"
LAB_EXECUTABLE="${RELU2_PYTHON:-${LAB_PYTHON:-$LAB_ROOT/venv/bin/python}}"
if [[ ! -x "$LAB_EXECUTABLE" ]]; then
  echo 'Run bash scripts/setup.sh first, or explicitly set RELU2_PYTHON.' >&2
  exit 1
fi
printf 'Python: %s\n' "$LAB_EXECUTABLE"
exec "$LAB_EXECUTABLE" -m context_sweep.run --preset configs/sweep_50m.json \
  --data "${RELU2_DATA:-$LAB_ROOT/data/fineweb_100m}" \
  --output runs/muon_output_norm_kda_50m "$@"

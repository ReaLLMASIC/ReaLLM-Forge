#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# An equal LR search budget for both variants, not selection based on test data.
# Default: THREE matched pairs, 10,000 updates EACH. Override LRS/STEPS explicitly.
read -r -a lrs <<< "${LRS:-0.0005 0.0015 0.003}"
ROOT="${OUT_ROOT:-runs/lr_sweep}"
for lr in "${lrs[@]}"; do
  OUT="$ROOT/lr_$lr" LR_GPT="$lr" LR_NGPT="$lr" bash scripts/run_4090.sh "$@"
done

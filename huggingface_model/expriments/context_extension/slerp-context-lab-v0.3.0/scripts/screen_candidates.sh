#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
case "${1:-both}" in
  both) models=(hunyuan minicpm) ;;
  hunyuan|minicpm) models=("$1") ;;
  *) echo 'Usage: bash scripts/screen_candidates.sh [both|hunyuan|minicpm]' >&2; exit 2 ;;
esac
inputs=()
for model in "${models[@]}"; do
  bash run.sh download --config "configs/$model-native.json"
  bash run.sh parity --config "configs/$model-local.json" --out "reports/$model-parity.json" --device cuda
  output="reports/screen/$model-native.jsonl"
  [[ ! -e "$output" ]] || { echo "Output exists: $output. Use explicit CLI commands for a new screen." >&2; exit 1; }
  bash run.sh evaluate --config "configs/$model-native.json" --out "$output" --device cuda     --lengths 4096 8192 16384 32768 --samples "${SAMPLES:-2}" --facts 8     --tasks recall multi trace --positions .05 .5 .9
  inputs+=("$output")
done
bash run.sh assess "${inputs[@]}" --out reports/screen/assessment.json
bash run.sh plot "${inputs[@]}" --out reports/screen/figures

#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
model="${1:-hunyuan}"
[[ "$model" == hunyuan || "$model" == minicpm ]] || { echo 'Choose hunyuan or minicpm' >&2; exit 2; }
inputs=()
for method in local nlerp slerp; do
  cfg="configs/$model-$method.json"
  run="runs/$model-$method-seed17"
  # Every arm is profiled with its actual settings before its longer run.
  if [[ ! -f "$run/TRAINING_COMPLETE.json" ]]; then
    bash run.sh profile --config "$cfg" --length 4096 --steps 3 --out "reports/profile-$model-$method-4096.json" --device cuda
  fi
  extra=()
  [[ ! -f "$run/latest.json" ]] || extra+=(--resume)
  bash run.sh train --config "$cfg" --out "$run" --device cuda "${extra[@]}"
  output="reports/pilot/$model-$method.jsonl"
  if [[ ! -e "$output" ]]; then
    bash run.sh evaluate --run "$run" --out "$output" --device cuda       --lengths 4096 8192 16384 32768 --samples "${SAMPLES:-4}" --facts 1 8 32
  fi
  bash run.sh assess "$output" --out "${output%.jsonl}.check.json"
  inputs+=("$output")
  if [[ "${FINALIZE_AFTER_EVAL:-0}" == 1 ]]; then bash run.sh finalize --run "$run"; fi
done
bash run.sh plot "${inputs[@]}" --out "reports/pilot/$model-figures"
bash run.sh assess "${inputs[@]}" --out "reports/pilot/$model-assessment.json"

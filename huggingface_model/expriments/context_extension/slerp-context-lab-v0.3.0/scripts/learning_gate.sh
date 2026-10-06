#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
inputs=()
for method in local nlerp slerp; do
  cfg="configs/gate-$method.json"
  run="runs/gate-$method-seed17"
  if [[ ! -f "$run/TRAINING_COMPLETE.json" ]]; then
    bash run.sh profile --config "$cfg" --length 768 --steps 3 --out "reports/gate-profile-$method.json" --device cuda
  fi
  extra=()
  [[ ! -f "$run/latest.json" ]] || extra+=(--resume)
  bash run.sh train --config "$cfg" --out "$run" --device cuda "${extra[@]}"
  output="reports/gate/$method.jsonl"
  if [[ ! -f "$output" ]]; then
    bash run.sh evaluate --run "$run" --out "$output" --device cuda       --lengths 768 --samples 32 --facts 2 --positions .05 --tasks recall
  fi
  bash run.sh assess "$output" --out "${output%.jsonl}.check.json"
  inputs+=("$output")
done
bash run.sh plot "${inputs[@]}" --out reports/gate/figures
bash run.sh assess "${inputs[@]}" --out reports/gate/comparison.json
# A diagnostic hurdle, not a significance test or a claimed benchmark result.
bash run.sh assess reports/gate/slerp.jsonl --out reports/gate/slerp-pass.json --threshold "${GATE_THRESHOLD:-0.8}"
echo 'Gate passed. Inspect the local/NLERP comparison before starting the larger pilot.'
echo 'To release optimizer space after inspection: bash scripts/finalize_gates.sh'

#!/usr/bin/env bash
# Full-weight shared warmup, then matched memory arms; validation only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$SLERP_PACKAGE_ROOT"
config="${1:-configs/summary-smol.json}"
warmup="${2:-configs/summary-smol-warmup.json}"
profiles="${3:-reports/summary-profiles}"
tag="${4:-summary-smol-seed17}"
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use a simple run tag.' >&2; exit 1; }
read -r -a methods <<< "${METHODS:-local nlerp slerp rmt delta}"
"$SLERP_PYTHON" scripts/check_summary_profiles.py "$config" "$warmup" "$profiles" "${methods[@]}"
results="reports/$tag";mkdir -p "$results"
shared="runs/$tag-shared"
if [[ -f "$shared/summary-run.json" ]]; then
  bash summary.sh train --config "$warmup" --out "$shared" --device cuda --resume
else
  bash summary.sh train --config "$warmup" --out "$shared" --device cuda
fi
for method in native_full head tail head_tail rolling map_reduce; do
  if [[ ! -f "$results/$method.jsonl" ]]; then
    bash summary.sh evaluate --config "$config" --method "$method" --checkpoint "$shared" --out "$results/$method.jsonl" --device cuda
  fi
done
for method in "${methods[@]}"; do
  run="runs/$tag-$method"
  if [[ -f "$run/summary-run.json" ]]; then
    bash summary.sh train --config "$config" --method "$method" --out "$run" --resume --device cuda
  else
    bash summary.sh train --config "$config" --method "$method" --out "$run" --initialize "$shared" --device cuda
  fi
  if [[ ! -f "$results/$method.jsonl" ]]; then
    bash summary.sh evaluate --config "$config" --method "$method" --checkpoint "$run" --out "$results/$method.jsonl" --device cuda
  fi
  # Optional explicit disk tradeoff: weights retained, optimizer/resume removed.
  if [[ "${FINALIZE_AFTER_EVAL:-0}" == 1 ]]; then
    "$SLERP_PYTHON" -c 'import sys; from slerp_context.summary.report import completed_rows; completed_rows([sys.argv[1]])' "$results/$method.jsonl"
    bash summary.sh finalize "$run"
  fi
done
if [[ ! -d "$results/plots" ]]; then
  bash summary.sh report "$results"/*.jsonl --out "$results/plots"
fi
echo "Validation results: $results/plots/summary.csv"
echo 'Choose settings using validation, then evaluate the fixed methods on --split test.'

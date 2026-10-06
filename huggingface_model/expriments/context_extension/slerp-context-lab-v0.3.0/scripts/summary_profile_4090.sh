#!/usr/bin/env bash
# Run before training. Each method includes backward AND AdamW allocation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$SLERP_PACKAGE_ROOT"
config="${1:-configs/summary-smol.json}"
warmup="${2:-configs/summary-smol-warmup.json}"
profiles="${3:-reports/summary-profiles}"
mkdir -p "$profiles"
warm_length="$("$SLERP_PYTHON" -c 'import sys; from slerp_context.summary.config import Study; print(Study.load(sys.argv[1]).train_source_limit)' "$warmup")"
long_length="$("$SLERP_PYTHON" -c 'import sys; from slerp_context.summary.config import Study; print(Study.load(sys.argv[1]).train_source_limit)' "$config")"
bash summary.sh profile --config "$warmup" --method native --source-tokens "$warm_length" --device cuda --out "$profiles/native.json"
read -r -a methods <<< "${METHODS:-local nlerp slerp rmt delta}"
for method in "${methods[@]}"; do
  bash summary.sh profile --config "$config" --method "$method" --source-tokens "$long_length" --device cuda --out "$profiles/$method.json"
done
echo 'Profiles passed. Review timings before starting scripts/summary_compare_4090.sh.'

#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash cjk_ICL/run_icl_sweep.sh
#
# Useful overrides:
#   MAX_EXAMPLES=12 FORCE=1 bash cjk_ICL/run_icl_sweep.sh
#   FAMILIES="tiktoken byte ipa" SHOT_COUNTS="0 1 3" bash cjk_ICL/run_icl_sweep.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-cjk_ICL/config.example.json}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FORCE="${FORCE:-0}"
FAMILIES="${FAMILIES:-}"
SHOT_COUNTS="${SHOT_COUNTS:-}"

args=(--config "$CONFIG" --device "$DEVICE" --dtype "$DTYPE")

if [[ -n "$MAX_EXAMPLES" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi
if [[ -n "$FAMILIES" ]]; then
  # shellcheck disable=SC2206
  family_args=($FAMILIES)
  args+=(--families "${family_args[@]}")
fi
if [[ -n "$SHOT_COUNTS" ]]; then
  # shellcheck disable=SC2206
  shot_args=($SHOT_COUNTS)
  args+=(--shot-counts "${shot_args[@]}")
fi

python cjk_ICL/icl_pipeline.py "${args[@]}"

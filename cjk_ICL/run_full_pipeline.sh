#!/usr/bin/env bash
set -euo pipefail

# Run the full CJK ICL pipeline from repo root:
#   bash cjk_ICL/run_full_pipeline.sh
#
# Useful smoke run:
#   SMOKE=1 FORCE=1 DEVICE=cpu DTYPE=float32 bash cjk_ICL/run_full_pipeline.sh
#
# Useful overrides:
#   FAMILIES="tiktoken byte ipa" SHOT_COUNTS="0 1 3" DEVICE=cuda DTYPE=bfloat16 bash cjk_ICL/run_full_pipeline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-cjk_translation}"
BENCHMARK_CSV="${BENCHMARK_CSV:-benchmarks_easy.csv}"
DATASETS="${DATASETS:-data/flores200-res/cjk_sentence_pairs.json data/ntrex/data/cjk_sentence_pairs.json}"
CONFIG="${CONFIG:-cjk_ICL/config.example.json}"
BUILD_SOURCE_DATA="${BUILD_SOURCE_DATA:-1}"
FORCE_SOURCE_DATA="${FORCE_SOURCE_DATA:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  FAMILIES="${FAMILIES:-tiktoken byte ipa}"
  MAX_EXAMPLES="${MAX_EXAMPLES:-12}"
  SHOT_COUNTS="${SHOT_COUNTS:-0 1}"
  DEVICE="${DEVICE:-cpu}"
  DTYPE="${DTYPE:-float32}"
else
  FAMILIES="${FAMILIES:-tiktoken byte ipa}"
  SHOT_COUNTS="${SHOT_COUNTS:-0 1 3}"
  DEVICE="${DEVICE:-cuda}"
  DTYPE="${DTYPE:-bfloat16}"
fi
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FORCE="${FORCE:-0}"

force_arg=()
if [[ "$FORCE" == "1" ]]; then
  force_arg=(--force)
fi

if [[ "$BUILD_SOURCE_DATA" == "1" ]]; then
  echo "Preparing source CJK pair files..."
  FORCE_SOURCE_DATA="$FORCE_SOURCE_DATA" SKIP_DOWNLOAD="$SKIP_DOWNLOAD" \
    bash data/flores200-res/build_cjk_pairs.sh
  FORCE_SOURCE_DATA="$FORCE_SOURCE_DATA" SKIP_DOWNLOAD="$SKIP_DOWNLOAD" \
    bash data/ntrex/build_cjk_pairs.sh
fi

echo "Preparing CJK canonical records..."
# shellcheck disable=SC2206
dataset_args=($DATASETS)
python3 cjk_translation_pipeline.py prepare-cjk-translation \
  --datasets "${dataset_args[@]}" \
  --out-dir "$DATA_DIR" \
  "${force_arg[@]}"

echo "Rendering tokenizer task files..."
# shellcheck disable=SC2206
family_args=($FAMILIES)
for family in "${family_args[@]}"; do
  python3 cjk_translation_pipeline.py render-cjk-translation \
    --data-dir "$DATA_DIR" \
    --tokenizer-family "$family" \
    "${force_arg[@]}"
done

echo "Adapting held-out benchmark..."
python3 cjk_translation_pipeline.py adapt-cjk-benchmark \
  --benchmark "$BENCHMARK_CSV" \
  --out "$DATA_DIR/benchmark_adapted/benchmarks_easy.jsonl"

if [[ ! -f "$DATA_DIR/selected_base_checkpoints.json" ]]; then
  echo "Missing checkpoint manifest: $DATA_DIR/selected_base_checkpoints.json" >&2
  echo "Run discover/select checkpoint preparation before running ICL." >&2
  exit 1
fi

echo "Running CJK ICL sweep..."
icl_args=(
  --config "$CONFIG"
  --families "${family_args[@]}"
  --device "$DEVICE"
  --dtype "$DTYPE"
)

if [[ -n "$SHOT_COUNTS" ]]; then
  # shellcheck disable=SC2206
  shot_args=($SHOT_COUNTS)
  icl_args+=(--shot-counts "${shot_args[@]}")
fi
if [[ -n "$MAX_EXAMPLES" ]]; then
  icl_args+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$FORCE" == "1" ]]; then
  icl_args+=(--force)
fi

python3 cjk_ICL/icl_pipeline.py "${icl_args[@]}"

echo
echo "CJK ICL pipeline complete."
echo "Scores: cjk_ICL/runs/all_icl_scores.csv"

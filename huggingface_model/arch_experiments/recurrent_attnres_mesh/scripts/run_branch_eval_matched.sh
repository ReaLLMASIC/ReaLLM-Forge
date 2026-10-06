#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_accelerate
require_single_gpu_process

DATASET="${DATASET:-roneneldan/TinyStories}"
TOKENIZER="${TOKENIZER:-gpt2}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/runs/${HARDWARE_PROFILE}_branch_eval_matched_$(run_stamp)_$$}"

# Each point performs eight attention and eight FFN branch evaluations per
# token. Router/readout costs and parameter counts still differ across points.
CASES=("1:1:8" "2:2:4" "4:4:2")
for case_spec in "${CASES[@]}"; do
  IFS=: read -r attention_modules ffn_modules iterations <<<"$case_spec"
  run_name="a${attention_modules}_f${ffn_modules}_s${iterations}"
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes 1 \
    "$MESH_SCRIPT" \
    --mode train \
    --hardware-profile "$HARDWARE_PROFILE" \
    --dataset "$DATASET" \
    --tokenizer "$TOKENIZER" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --num-attention-modules "$attention_modules" \
    --num-ffn-modules "$ffn_modules" \
    --num-iterations "$iterations" \
    --output-dir "$OUTPUT_ROOT/$run_name" \
    --no-save-model \
    "$@"
done

echo "Branch-evaluation-matched runs: $OUTPUT_ROOT"

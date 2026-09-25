#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_accelerate
require_single_gpu_process

DATASET="${DATASET:-roneneldan/TinyStories}"
TOKENIZER="${TOKENIZER:-gpt2}"
MAX_STEPS="${MAX_STEPS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/${HARDWARE_PROFILE}_tied_grid_$(run_stamp)_$$}"
if [[ "${FULL_GRID:-0}" == "1" ]]; then
  ATTENTION_MODULES="${ATTENTION_MODULES:-0,1,2,4}"
  FFN_MODULES="${FFN_MODULES:-0,1,2,4}"
else
  ATTENTION_MODULES="${ATTENTION_MODULES:-1,2,4}"
  FFN_MODULES="${FFN_MODULES:-1,2,4}"
fi
ITERATIONS="${ITERATIONS:-1,2,4,6,8}"

case "$HARDWARE_PROFILE" in
  a100-80gb|h100-sxm-80gb|h100-pcie-80gb)
    default_sweep_micro_batch_size=4
    default_sweep_gradient_accumulation_steps=4
    ;;
  rtx4090-24gb)
    default_sweep_micro_batch_size=2
    default_sweep_gradient_accumulation_steps=16
    ;;
  cuda-generic|portable)
    default_sweep_micro_batch_size=1
    default_sweep_gradient_accumulation_steps=32
    ;;
esac
SWEEP_MICRO_BATCH_SIZE="${SWEEP_MICRO_BATCH_SIZE:-$default_sweep_micro_batch_size}"
SWEEP_GRADIENT_ACCUMULATION_STEPS="${SWEEP_GRADIENT_ACCUMULATION_STEPS:-$default_sweep_gradient_accumulation_steps}"

exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode sweep \
  --hardware-profile "$HARDWARE_PROFILE" \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  --micro-batch-size "$SWEEP_MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$SWEEP_GRADIENT_ACCUMULATION_STEPS" \
  --sweep-attention-modules "$ATTENTION_MODULES" \
  --sweep-ffn-modules "$FFN_MODULES" \
  --sweep-iterations "$ITERATIONS" \
  --sweep-router-types attnres \
  --sweep-share-iteration-weights tied \
  --no-sweep-save-models \
  --sweep-stop-on-error \
  --no-sweep-stop-on-oom \
  "$@"

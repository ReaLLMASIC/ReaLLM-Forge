#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_accelerate
require_single_gpu_process

DATASET="${DATASET:-roneneldan/TinyStories}"
TOKENIZER="${TOKENIZER:-gpt2}"
MAX_STEPS="${MAX_STEPS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/${HARDWARE_PROFILE}_untied_grid_$(run_stamp)_$$}"
ATTENTION_MODULES="${ATTENTION_MODULES:-1,2,4}"
FFN_MODULES="${FFN_MODULES:-1,2,4}"
ITERATIONS="${ITERATIONS:-1,2,4,6}"

case "$HARDWARE_PROFILE" in
  a100-80gb|h100-sxm-80gb|h100-pcie-80gb)
    default_sweep_micro_batch_size=2
    default_sweep_gradient_accumulation_steps=8
    ;;
  rtx4090-24gb|cuda-generic|portable)
    default_sweep_micro_batch_size=1
    default_sweep_gradient_accumulation_steps=32
    ;;
esac
SWEEP_MICRO_BATCH_SIZE="${SWEEP_MICRO_BATCH_SIZE:-$default_sweep_micro_batch_size}"
SWEEP_GRADIENT_ACCUMULATION_STEPS="${SWEEP_GRADIENT_ACCUMULATION_STEPS:-$default_sweep_gradient_accumulation_steps}"

# Untied depth multiplies branch parameters, so this launcher uses a
# profile-specific, memory-safer batch. GPU profiles retain 32,768
# tokens/update; portable remains a smaller CPU diagnostic.
exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode sweep \
  --hardware-profile "$HARDWARE_PROFILE" \
  --dataset "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  --micro-batch-size "$SWEEP_MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$SWEEP_GRADIENT_ACCUMULATION_STEPS" \
  --gradient-checkpointing \
  --sweep-attention-modules "$ATTENTION_MODULES" \
  --sweep-ffn-modules "$FFN_MODULES" \
  --sweep-iterations "$ITERATIONS" \
  --sweep-router-types attnres \
  --sweep-share-iteration-weights untied \
  --no-sweep-save-models \
  --sweep-stop-on-error \
  --no-sweep-stop-on-oom \
  "$@"

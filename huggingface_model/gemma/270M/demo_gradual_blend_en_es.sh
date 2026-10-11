#!/usr/bin/env bash
set -euo pipefail

# Demo: EN->ES gradual blend fine-tuning for Gemma 270M
#
# This script shows the full staged flow:
#   1) obtain / choose a checkpoint
#   2) optional Softmax warmup stage
#   3) gradual blend stage with alpha annealing
#
# It also includes commented alternatives for:
#   - pure Softmax baseline
#   - summed scores baseline (Softmax + ReLU variant)
#
# Usage:
#   bash huggingface_model/gemma/270M/demo_gradual_blend_en_es.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ----------------------------
# Common configuration
# ----------------------------
BASE_MODEL="${BASE_MODEL:-google/gemma-3-270m}"
DATASET_NAME="${DATASET_NAME:-Helsinki-NLP/opus-100}"
DATASET_CONFIG="${DATASET_CONFIG:-en-es}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:10%]}"
SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-es}"
SOURCE_LANG_NAME="${SOURCE_LANG_NAME:-English}"
TARGET_LANG_NAME="${TARGET_LANG_NAME:-Spanish}"

# Training controls (override through env vars if desired)
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
GRADUAL_STEPS="${GRADUAL_STEPS:-4000}"
POST_ZERO_STEPS="${POST_ZERO_STEPS:-500}"
SAMPLE_FREQ="${SAMPLE_FREQ:-500}"

# ReLU branch options for gradual/sum variants
ATTN_ACTIVATION="${ATTN_ACTIVATION:-relu2max}"   # relumax | relu2max
ACTIVATION_DIVISOR="${ACTIVATION_DIVISOR:-256.0}"

# Output directories
RUN_ROOT="${RUN_ROOT:-./runs/gemma270_demo_en_es}"
SOFTMAX_STAGE_DIR="${SOFTMAX_STAGE_DIR:-${RUN_ROOT}/stage1_softmax}"
GRADUAL_STAGE_DIR="${GRADUAL_STAGE_DIR:-${RUN_ROOT}/stage2_gradual_${ATTN_ACTIVATION}}"
SUM_BASELINE_DIR="${SUM_BASELINE_DIR:-${RUN_ROOT}/baseline_sum_${ATTN_ACTIVATION}}"
SOFTMAX_BASELINE_DIR="${SOFTMAX_BASELINE_DIR:-${RUN_ROOT}/baseline_softmax}"

mkdir -p "${RUN_ROOT}"

echo "== Stage 1 (optional): Softmax warmup from ${BASE_MODEL} =="
python huggingface_model/gemma/270M/finetune.py \
  --model_name "${BASE_MODEL}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config "${DATASET_CONFIG}" \
  --dataset_split "${DATASET_SPLIT}" \
  --source_lang "${SOURCE_LANG}" \
  --target_lang "${TARGET_LANG}" \
  --source_lang_name "${SOURCE_LANG_NAME}" \
  --target_lang_name "${TARGET_LANG_NAME}" \
  --output_dir "${SOFTMAX_STAGE_DIR}" \
  --total_iterations "${WARMUP_STEPS}" \
  --sample_frequency "${SAMPLE_FREQ}" \
  --attention_mode softmax

echo "== Stage 2: Gradual blend from Softmax checkpoint =="
python huggingface_model/gemma/270M/finetune.py \
  --model_name "${SOFTMAX_STAGE_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config "${DATASET_CONFIG}" \
  --dataset_split "${DATASET_SPLIT}" \
  --source_lang "${SOURCE_LANG}" \
  --target_lang "${TARGET_LANG}" \
  --source_lang_name "${SOURCE_LANG_NAME}" \
  --target_lang_name "${TARGET_LANG_NAME}" \
  --output_dir "${GRADUAL_STAGE_DIR}" \
  --total_iterations "${GRADUAL_STEPS}" \
  --sample_frequency "${SAMPLE_FREQ}" \
  --attention_mode gradual_blend \
  --attention_activation "${ATTN_ACTIVATION}" \
  --activation_divisor "${ACTIVATION_DIVISOR}" \
  --alpha_start 1.0 \
  --alpha_end 0.0 \
  --post_zero_steps "${POST_ZERO_STEPS}" \
  --blend_output_norm

echo "== Done. Final gradual checkpoint: ${GRADUAL_STAGE_DIR} =="

cat <<'EOF'

Other options (copy/paste if needed):

1) Softmax-only baseline
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_demo_en_es/baseline_softmax \
  --total_iterations 4000 \
  --sample_frequency 500 \
  --attention_mode softmax

2) Sum baseline (Softmax + relu variant)
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_demo_en_es/baseline_sum_relu2max \
  --total_iterations 4000 \
  --sample_frequency 500 \
  --attention_mode sum \
  --attention_activation relu2max \
  --activation_divisor 256.0

3) Plot validation loss curves
python huggingface_model/gemma/270M/plot_validation_loss.py \
  --run "gradual=./runs/gemma270_demo_en_es/stage2_gradual_relu2max" \
  --run "softmax=./runs/gemma270_demo_en_es/baseline_softmax" \
  --run "sum=./runs/gemma270_demo_en_es/baseline_sum_relu2max" \
  --output ./runs/gemma270_demo_en_es/val_loss_compare.png

4) Benchmark EN->ES translation
python huggingface_model/gemma/270M/benchmark_en_es_translation.py \
  --model_name ./runs/gemma270_demo_en_es/stage2_gradual_relu2max \
  --dataset_config en-es \
  --dataset_split "train[10%:11%]" \
  --num_samples 200

EOF

#!/bin/bash
# Probe pretrained HuggingFace instruction models for special Wv/Wo head matrices.
# Defaults target:
#   - google/gemma-3-270m-it
#   - HuggingFaceTB/SmolLM2-135M-Instruct (the SmolLM URL requested by the demo)
# The Gemma repository may require accepting Google's Gemma license and having an
# authenticated Hugging Face token available in the environment.

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUT_ROOT="${OUT_ROOT:-out/huggingface_special_matrix_probe}"
ROTATION_MIN_DEG="${ROTATION_MIN_DEG:-0}"
ROTATION_MAX_DEG="${ROTATION_MAX_DEG:-180}"
ROTATION_STEP_DEG="${ROTATION_STEP_DEG:-5}"
MAX_LAYERS="${MAX_LAYERS:-}"
MODEL_LIST="${MODELS:-google/gemma-3-270m-it HuggingFaceTB/SmolLM2-135M-Instruct}"
read -r -a MODEL_ARRAY <<< "${MODEL_LIST}"

mkdir -p "${OUT_ROOT}"

for MODEL in "${MODEL_ARRAY[@]}"; do
  SAFE_NAME="${MODEL//\//__}"
  MODEL_OUT_DIR="${OUT_ROOT}/${SAFE_NAME}"
  EXTRA_ARGS=()
  if [ -n "${MAX_LAYERS}" ]; then
    EXTRA_ARGS+=(--max-layers "${MAX_LAYERS}")
  fi
  if [[ "${MODEL}" == google/gemma-* ]]; then
    EXTRA_ARGS+=(--trust-remote-code)
  fi

  echo "=== Probing ${MODEL} ==="
  python3 analysis/identity/head_special_matrix_probe.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --outdir "${MODEL_OUT_DIR}" \
    --rotation-min-deg "${ROTATION_MIN_DEG}" \
    --rotation-max-deg "${ROTATION_MAX_DEG}" \
    --rotation-step-deg "${ROTATION_STEP_DEG}" \
    "${EXTRA_ARGS[@]}"
done

cat <<MSG
Done.
Reports are under ${OUT_ROOT}.
Each model directory contains head_special_matrix_scores.csv plus heatmaps for
identity, negative identity, orthogonal/involution/projection/symmetric/skew
metrics, and ${ROTATION_MIN_DEG}..${ROTATION_MAX_DEG} degree rotation targets at
${ROTATION_STEP_DEG} degree increments.
MSG

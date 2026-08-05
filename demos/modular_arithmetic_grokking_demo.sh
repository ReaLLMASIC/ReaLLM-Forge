#!/bin/bash
# Train a small nanoGPT model on modular addition and probe learned attention
# heads for identity/projection/skew/rotation-like Wv/Wo structure.

set -euo pipefail

MODULUS="${MODULUS:-113}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.3}"
TRAIN_REPEATS="${TRAIN_REPEATS:-200}"
VAL_REPEATS="${VAL_REPEATS:-20}"
MAX_ITERS="${MAX_ITERS:-20000}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
OUT_DIR="${OUT_DIR:-out/modular_arithmetic_grokking_p${MODULUS}}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-${OUT_DIR}/head_special_matrix_probe}"
DATA_DIR="data/modular_arithmetic"
CKPT_PATH="${OUT_DIR}/ckpt.pt"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

echo "=== Step 1: Prepare modular addition dataset ==="
python3 "${DATA_DIR}/prepare.py" \
  --out-dir "${DATA_DIR}" \
  --modulus "${MODULUS}" \
  --train-fraction "${TRAIN_FRACTION}" \
  --train-repeats "${TRAIN_REPEATS}" \
  --val-repeats "${VAL_REPEATS}"

echo "=== Step 2: Train grokking-sized nanoGPT model ==="
python3 train.py \
  --dataset modular_arithmetic \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --block_size 32 \
  --batch_size 256 \
  --n_layer 1 \
  --n_head 4 \
  --n_embd 128 \
  --dropout 0.0 \
  --bias \
  --max_iters "${MAX_ITERS}" \
  --eval_interval 500 \
  --eval_iters 100 \
  --learning_rate 1e-3 \
  --weight_decay 1.0 \
  --warmup_iters 100 \
  --decay_lr \
  --min_lr 1e-5 \
  --always_save_checkpoint \
  --only_save_checkpoint_at_end \
  --no-compile

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

echo "=== Step 3: Probe Wv/Wo special matrices, including 0..180 degree rotations ==="
python3 analysis/identity/head_special_matrix_probe.py \
  --ckpt "${CKPT_PATH}" \
  --device "${DEVICE%%:*}" \
  --dtype "float32" \
  --outdir "${PROBE_OUT_DIR}" \
  --rotation-min-deg 0 \
  --rotation-max-deg 180 \
  --rotation-step-deg 5

cat <<MSG
Done.
Dataset artifacts: ${DATA_DIR}/train.bin, ${DATA_DIR}/val.bin, ${DATA_DIR}/meta.pkl
Training output: ${OUT_DIR}
Special matrix probe report: ${PROBE_OUT_DIR}
MSG

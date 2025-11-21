#!/bin/bash
# snap_to_grid_demo.sh
# Demonstrates training and sampling with snap-to-grid projections.

set -euo pipefail

DATASET="${DATASET:-shakespeare_char}"
DATA_DIR="data/${DATASET}"
OUT_DIR="${OUT_DIR:-out/snap_to_grid_demo}"
SNAP_SIZES_STR="${SNAP_SIZES:-"8 32"}"
IFS=' ' read -r -a SNAP_SIZES <<< "${SNAP_SIZES_STR}"

# Model hyperparameters (override via env vars, e.g. `N_HEAD=3 ./snap_to_grid_demo.sh`).
N_LAYER=${N_LAYER:-4}
N_HEAD=${N_HEAD:-4}
N_EMBD=${N_EMBD:-128}

if (( N_EMBD % N_HEAD != 0 )); then
  echo "error: N_EMBD (${N_EMBD}) must be divisible by N_HEAD (${N_HEAD})." >&2
  echo "update N_EMBD or N_HEAD before running the demo." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=== Step 1: Ensure the ${DATASET} dataset is prepared ==="
if [ ! -f "${DATA_DIR}/train.bin" ] || [ ! -f "${DATA_DIR}/val.bin" ]; then
  pushd "${DATA_DIR}" > /dev/null
  if [ ! -f "input.txt" ]; then
    echo "Downloading Shakespeare corpus..."
    bash get_dataset.sh
  fi
  echo "Tokenizing dataset with tiktoken encoder..."
  python3 prepare.py -t input.txt --method tiktoken
  popd > /dev/null
else
  echo "Found existing tokenized dataset artifacts."
fi

CKPT_PATH="${OUT_DIR}/ckpt.pt"

cat <<CONFIG
=== Step 2: Train a tiny model with snap-to-grid enabled ===
 - output directory: ${OUT_DIR}
 - snap-to-grid sizes evaluated: ${SNAP_SIZES[*]}
CONFIG

python3 train.py \
  --dataset "${DATASET}" \
  --out_dir "${OUT_DIR}" \
  --block_size 128 \
  --batch_size 12 \
  --n_layer "${N_LAYER}" \
  --n_head "${N_HEAD}" \
  --n_embd "${N_EMBD}" \
  --max_iters 200 \
  --eval_interval 100 \
  --eval_iters 50 \
  --log_interval 10 \
  --learning_rate 3e-4 \
  --enable_snap_to_grid \
  --snap_to_grid_sizes "${SNAP_SIZES[@]}" \
  --snap_to_grid_components both

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

cat <<CONFIG
=== Step 3: Evaluate validation loss with snap-to-grid registries ===
 - checkpoint: ${CKPT_PATH}
 - snap-to-grid sizes: ${SNAP_SIZES[*]}
CONFIG

python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --init_from resume \
  --eval_only \
  --eval_dataset "${DATASET}" \
  --eval_iters 50 \
  --enable_snap_to_grid \
  --snap_to_grid_sizes "${SNAP_SIZES[@]}"

cat <<CONFIG
=== Step 4: Generate baseline samples (snap-to-grid disabled) ===
CONFIG

python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --init_from resume \
  --start "ROMEO: " \
  --num_samples 1 \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 200 \
  --seed 1337

cat <<CONFIG
=== Step 5: Generate samples with snap-to-grid enabled ===
 - snap-to-grid sizes: ${SNAP_SIZES[*]}
CONFIG

python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --init_from resume \
  --start "ROMEO: " \
  --num_samples 1 \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 200 \
  --seed 1337 \
  --enable_snap_to_grid \
  --snap_to_grid_sizes "${SNAP_SIZES[@]}"

cat <<MSG
Demo complete. Snap-to-grid registries and logs are stored under ${OUT_DIR}/snap_to_grid.
Check tensorboard logs for the snap_to_grid/val_loss_size_* series to compare validation performance.
MSG

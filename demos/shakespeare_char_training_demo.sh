#!/bin/bash
# shakespeare_char_training_demo.sh
# Demonstrates training a GPT-2 124M-style configuration on the minipile dataset
# with rotary embeddings, QK norm, QK norm scale, and peri-layer normalization,
# then inspects the resulting checkpoint with the regex explorer.

set -euo pipefail

DATA_DIR="data/minipile"
OUT_DIR="out/minipile_gpt2_124m_demo"
HIST_DIR="${OUT_DIR}/regex_histograms"
CKPT_PATH="${OUT_DIR}/ckpt.pt"

mkdir -p "${DATA_DIR}"

echo "=== Step 1: Prepare the minipile dataset ==="
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "${OUT_DIR}"

echo "=== Step 2: Train a GPT-2 124M-style model on minipile ==="
python3 train.py \
  --dataset minipile \
  --out_dir "${OUT_DIR}" \
  --block_size 1024 \
  --batch_size 12 \
  --n_layer 12 \
  --n_head 12 \
  --n_embd 768 \
  --max_iters 2000 \
  --eval_interval 200 \
  --eval_iters 200 \
  --learning_rate 6e-4 \
  --weight_decay 0.1 \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_peri_ln \
  --compile

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

echo "=== Step 3: Analyze attention projection weights with the regex explorer ==="
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.h\\.[0-9]+\\.attn\\.c_proj\\.weight" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 60

echo "=== Step 4: Analyze MLP projection weights with the regex explorer ==="
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.h\\.[0-9]+\\.mlp\\.c_proj\\.weight" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 60

echo "=== Step 5: Analyze token embeddings with the regex explorer ==="
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.wte\\.weight" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 60

cat <<MSG
Training complete. Checkpoints, logs, and histogram images live under ${OUT_DIR}.
Pairwise angle statistics are reported in degrees by default; pass --angle-units radians
when invoking the regex explorer to switch units.
MSG

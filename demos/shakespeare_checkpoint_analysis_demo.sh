#!/bin/bash
# demos/shakespeare_checkpoint_analysis_demo.sh
# Demonstrates training on the shakespeare_char dataset and exploring checkpoints.

set -euo pipefail

SKIP_TRAINING="${1:-no}"
DATA_DIR="data/shakespeare_char"
OUT_DIR="out_shakespeare_checkpoint_demo"
HIST_DIR="${OUT_DIR}/regex_histograms"
CKPT_PATH="${OUT_DIR}/ckpt.pt"

# Ensure the dataset is available.
pushd "${DATA_DIR}" > /dev/null
bash get_dataset.sh
popd > /dev/null

if [[ "$SKIP_TRAINING" = "no" ]]; then
  # Train a compact model to produce a checkpoint for analysis.
  rm -rf "${OUT_DIR}"
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "${OUT_DIR}" \
    --max_iters 1000 \
    --attention_variant infinite \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --use_concat_heads \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size 64 \
    --batch_size 64 \
    --n_layer 10 \
    --n_head 6 \
    --n_embd 384 \
    --eval_interval 100 \
    --log_interval 10 \
    --compile
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Expected checkpoint not found at ${CKPT_PATH}" >&2
  exit 1
fi

# Use the interactive explorer to inspect embedding statistics.
echo "\nRunning checkpoint explorer to inspect transformer.wte.weight statistics..."
python3 analysis/checkpoint_analysis/checkpoint_explorer.py "${CKPT_PATH}" <<'EOINPUT'
0
0
0
4

b
b
b
q
EOINPUT

# Use the regex explorer to summarize attention projection weights and save histograms.
echo "\nRunning checkpoint regex explorer for attention projection weights..."
python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.h\\.[0-9]+\\.attn\\.(c_attn_(q|k|v)|c_proj)\\.weight" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 40

python3 analysis/checkpoint_analysis/checkpoint_regex_explorer.py \
  "${CKPT_PATH}" \
  "transformer\\.h\\.[0-9]+\\.mlp\\.(c_fc|c_proj)\\.weight" \
  --histogram-dir "${HIST_DIR}" \
  --histogram-bins 40

echo "\nHistogram images saved under ${HIST_DIR}"

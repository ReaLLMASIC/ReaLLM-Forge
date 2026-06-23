#!/bin/bash
# demos/lm_head_vocab_histogram_demo.sh
# Quick demo: train a tiny Shakespeare-char model while logging a TensorBoard
# histogram of lm_head vocab-vector magnitudes over time.

set -euo pipefail

OUT_DIR="out/lm_head_vocab_hist_demo"
RUN_NAME="lm_head_vocab_hist_demo"

echo "=== Training tiny model with lm_head vocab histogram logging enabled ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "${OUT_DIR}" \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 128 \
  --block_size 128 \
  --batch_size 32 \
  --max_iters 200 \
  --eval_interval 50 \
  --eval_iters 20 \
  --log_interval 10 \
  --learning_rate 1e-3 \
  --tensorboard_log \
  --tensorboard_run_name "${RUN_NAME}" \
  --log_lm_head_vocab_hist \
  --log_lm_head_vocab_hist_interval 10 \
  --export_lm_head_vocab_hist_html \
  --lm_head_vocab_hist_html_path "${OUT_DIR}/lm_head_vocab_histogram.html"

cat <<MSG
Done.

To view the histogram animation in TensorBoard:
  tensorboard --logdir logs

Then open TensorBoard and navigate to:
  shakespeare_char/lm_head_vocab_vector_magnitude

Use the Histograms or Distributions tab and scrub over steps to animate the
change in lm_head vocab-vector magnitude distribution during training.

An interactive final-snapshot HTML is also written to:
  ${OUT_DIR}/lm_head_vocab_histogram.html
MSG

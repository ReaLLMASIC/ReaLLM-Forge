#!/usr/bin/env bash
# Generic integer CSV regular multicontext demo:
# 1) split a CSV into one integer-range dataset per column
# 2) train regular multicontext with a separate vocabulary per column
# 3) sample from a CSV prompt and write timestamped CSV continuations

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CSV_INPUT="${1:-data/csv_mc_int/roomba_integer.csv}"
OUTPUT_ROOT="${CSV_MC_OUTPUT_ROOT:-csv_mc_int}"
OUT_DIR="${CSV_MC_OUT_DIR:-out/csv_mc_int}"
MAX_ITERS="${CSV_MC_MAX_ITERS:-1000}"

# Override or extend these ranges for your own CSV. Every value is checked
# get_dataset and scripts here

mapfile -t DATASETS < <(python3 - <<PY
import json
from pathlib import Path
manifest = json.loads(Path('data/$OUTPUT_ROOT/manifest.json').read_text())
for dataset in manifest['multicontext_datasets']:
    print(dataset)
PY
)

python3 train.py \
  --training_mode multicontext \
  --dataset "${DATASETS[0]}" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --n_layer 6 \
  --n_head 6 \
  --attention_variant infinite \
  --use_concat_heads \
  --n_qk_head_dim 200 \
  --n_v_head_dim 112 \
  --n_embd 128 \
  --block_size 100 \
  --batch_size 2 \
  --max_iters "$MAX_ITERS" \
  --eval_interval 50 \
  --eval_iters 10 \
  --learning_rate 1e-3 \
  --dropout 0.0 \
  --device "${CSV_MC_DEVICE:-cuda:0}" \
  --dtype bfloat16 \
  --optimizer muon \
  --weight_decay 0.0 \
  --softmax_variant_attn relu2max \
  --use_qk_norm \
  --use_qk_norm_scale \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --compile \
  --out_dir "$OUT_DIR"

python3 sample.py \
  --out_dir "$OUT_DIR" \
  --device "${CSV_MC_DEVICE:-cuda:0}" \
  --dtype bfloat16 \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_csv_input "$CSV_INPUT" \
  --multicontext_csv_output_dir "$OUT_DIR/csv_samples" \
  --max_new_tokens 10000 \
  --top_k 5 \
  --compile \
  --num_samples 1

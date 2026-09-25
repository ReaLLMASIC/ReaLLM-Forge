#!/usr/bin/env bash
# End-to-end smoke demo for simplified Hanzi radical-location multicontext:
# data prep -> tiny training run -> sampling -> bijective char-lane reconstruction.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
DATA_ROOT="data/simplified_hanzi_mc"
OUT_DIR="out/simplified_hanzi_mc_demo"

bash "${DATA_ROOT}/get_dataset.sh"

mapfile -t DATASETS < <(python3 - <<'PY'
import json
m=json.load(open('data/simplified_hanzi_mc/manifest.json', encoding='utf-8'))
print('\n'.join(m['multicontext_datasets']))
PY
)

python3 train.py \
  --training_mode multicontext \
  --dataset "data/simplified_hanzi_mc/char/char_simplified_hanzi_mc" \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --out_dir "${OUT_DIR}" \
  --eval_interval 250 \
  --eval_iters 100 \
  --log_interval 10 \
  --always_save_checkpoint \
  --max_iters "${MAX_ITERS:-10000}" \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_qk_norm \
  --use_qk_norm_scale \
  --batch_size 32 \
  --block_size 256 \
  --n_layer 10 \
  --n_head 3 \
  --n_embd 384 \
  --dropout 0.0 \
  --device "${DEVICE:-cuda:0}" \
  --no-compile

python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE:-cuda:0}" \
  --no-compile \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_start "明" "∅" "∅" "日" "月" "∅" "∅" "∅" "∅" "∅" "∅" "∅" \
  --max_new_tokens 16 \
  --top_k 1 \
  --num_samples 1 | tee "${OUT_DIR}/sample.txt"

# Deterministic reconstruction smoke test from the prepared char lane. For model
# output, save/generated the char-lane continuation and pass --char_file to this script.
python3 "${DATA_ROOT}/decode_multicontext_sample.py" --root "${DATA_ROOT}" | tee "${OUT_DIR}/decoded_reference.txt"

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
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --out_dir "${OUT_DIR}" \
  --eval_interval 5 \
  --eval_iters 2 \
  --log_interval 1 \
  --always_save_checkpoint \
  --max_iters "${MAX_ITERS:-20}" \
  --batch_size 4 \
  --block_size 8 \
  --n_layer 2 \
  --n_head 2 \
  --n_embd 64 \
  --dropout 0.0 \
  --device "${DEVICE:-cpu}" \
  --no-compile

python3 sample.py \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE:-cpu}" \
  --no-compile \
  --multicontext \
  --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_start "明" "∅" "日" "月" "∅" "∅" "∅" "∅" "∅" "∅" "∅" \
  --max_new_tokens 16 \
  --top_k 1 \
  --num_samples 1 | tee "${OUT_DIR}/sample.txt"

# Deterministic reconstruction smoke test from the prepared char lane. For model
# output, save/generated the char-lane continuation and pass --char_file to this script.
python3 "${DATA_ROOT}/decode_multicontext_sample.py" --root "${DATA_ROOT}" | tee "${OUT_DIR}/decoded_reference.txt"

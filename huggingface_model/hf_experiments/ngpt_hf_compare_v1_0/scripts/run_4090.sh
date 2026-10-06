#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ ! -f .venv/bin/activate ]] || source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
DATA="${DATA:-data/owt_200m}"
OUT="${OUT:-runs/owt_4090}"
if [[ ! -f "$DATA/metadata.json" ]]; then
  python compare_ngpt.py prepare --data "$DATA" \
    --dataset "${DATASET:-Skylion007/openwebtext}" --tokenizer "${TOKENIZER:-gpt2}" \
    --train-tokens "${TRAIN_TOKENS:-200000000}" --val-tokens "${VAL_TOKENS:-1000000}"
else
  echo "Reusing immutable prepared data: $DATA (metadata.json records dataset/tokenizer)."
fi
read -r -a seeds <<< "${SEEDS:-0}"
extra=()
[[ "${RESUME:-0}" != 1 ]] || extra+=(--resume)
[[ "${ACTIVATION_CHECKPOINTING:-0}" != 1 ]] || extra+=(--activation-checkpointing)
python compare_ngpt.py train --data "$DATA" --out "$OUT" \
  --width "${WIDTH:-512}" --layers "${LAYERS:-8}" --heads "${HEADS:-8}" \
  --context "${CONTEXT:-1024}" --steps "${STEPS:-10000}" \
  --batch-size "${BATCH_SIZE:-2}" --accumulation "${ACCUMULATION:-16}" \
  --lr-gpt "${LR_GPT:-0.003}" --lr-ngpt "${LR_NGPT:-0.003}" \
  --warmup-gpt "${WARMUP_GPT:-2000}" --seeds "${seeds[@]}" \
  --precision bf16 --weight-storage "${WEIGHT_STORAGE:-fp32}" "${extra[@]}" "$@"

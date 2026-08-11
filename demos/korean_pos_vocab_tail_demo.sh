#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Korean POS Factorized Tokenizer vs Baseline: Vocabulary Tail Test       "
echo " Zero-Shot Perplexity across 10 Hangul Syllable Frequency Deciles        "
echo "=========================================================================="

MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
DEVICE="${DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"

MC_DIR="./out_mc_korean_pos"
BASE_DIR="./out_baseline_korean_pos"
TRAIN_CORPUS="data/korean_pos_mc/input.txt"

if [[ ! -d "$MC_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Error: Checkpoint directories $MC_DIR or $BASE_DIR do not exist."
  exit 1
fi

python3 benchmarks/run_vocab_tail_perplexity.py \
  --mc_dir "$MC_DIR" \
  --base_dir "$BASE_DIR" \
  --train_corpus "$TRAIN_CORPUS" \
  --dataset_name "KETI-AIR/kor_hellaswag" \
  --split "validation" \
  --max_examples "$MAX_EXAMPLES" \
  --device "$DEVICE" \
  --output_json ./vocab_tail_perplexity_results.json

echo "=========================================================================="

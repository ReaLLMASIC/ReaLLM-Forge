#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Korean POS Factorized Tokenizer vs Baseline: Experiment 1 Robustness "
echo " Phonetic Slang & Typo Resilience Evaluation on 3k Checkpoints           "
echo "=========================================================================="

MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
CORRUPTION_RATE="${CORRUPTION_RATE:-0.8}"
DEVICE="${DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"

MC_DIR="./out_mc_korean_pos"
BASE_DIR="./out_baseline_korean_pos"

if [[ ! -d "$MC_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Error: Checkpoint directories $MC_DIR or $BASE_DIR do not exist."
  echo "Please ensure 3k checkpoints are present."
  exit 1
fi

python3 benchmarks/run_phonetic_slang_eval.py \
  --mc_dir "$MC_DIR" \
  --base_dir "$BASE_DIR" \
  --max_examples "$MAX_EXAMPLES" \
  --corruption_rate "$CORRUPTION_RATE" \
  --device "$DEVICE" \
  --output_json ./phonetic_slang_resilience_results.json

echo "=========================================================================="

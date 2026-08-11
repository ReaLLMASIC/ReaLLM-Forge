#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Starting Full Evaluation Runs for Phonetic Slang & Typo Resilience    "
echo "=========================================================================="

DEVICE="${DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"
MC_DIR="./out_mc_korean_pos"
BASE_DIR="./out_baseline_korean_pos"
CORRUPTION_RATE="${CORRUPTION_RATE:-0.8}"

echo "\n[Run 1/3] Ko-HellaSwag FULL Validation Split (10,042 examples)..."
python3 benchmarks/run_phonetic_slang_eval.py \
  --mc_dir "$MC_DIR" \
  --base_dir "$BASE_DIR" \
  --dataset_name "KETI-AIR/kor_hellaswag" \
  --split "validation" \
  --max_examples 0 \
  --corruption_rate "$CORRUPTION_RATE" \
  --device "$DEVICE" \
  --output_json ./full_ko_hellaswag_val_results.json

echo "\n[Run 2/3] Ko-HellaSwag FULL Test Split (10,003 examples)..."
python3 benchmarks/run_phonetic_slang_eval.py \
  --mc_dir "$MC_DIR" \
  --base_dir "$BASE_DIR" \
  --dataset_name "KETI-AIR/kor_hellaswag" \
  --split "test" \
  --max_examples 0 \
  --corruption_rate "$CORRUPTION_RATE" \
  --device "$DEVICE" \
  --output_json ./full_ko_hellaswag_test_results.json

echo "\n[Run 3/3] NSMC FULL Test Split (50,000 examples)..."
python3 benchmarks/run_phonetic_slang_eval.py \
  --mc_dir "$MC_DIR" \
  --base_dir "$BASE_DIR" \
  --dataset_name "Blpeng/nsmc" \
  --split "test" \
  --max_examples 0 \
  --corruption_rate "$CORRUPTION_RATE" \
  --device "$DEVICE" \
  --output_json ./full_nsmc_test_results.json

echo "=========================================================================="
echo " All Full Evaluation Runs Completed!                                     "
echo "=========================================================================="

#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Running All Evaluations for 3, 5, and 10 Epoch Checkpoints              "
echo "=========================================================================="

DEVICE="${DEVICE:-cuda:0}"
MC_DIR="./out_mc_korean_pos"
BASE_DIR="./out_baseline_korean_pos"

epochs=(3 5 10)
ckpts=(3472 5786 11572)

for i in "${!epochs[@]}"; do
  ep="${epochs[$i]}"
  ckpt="${ckpts[$i]}.pt"
  mc_ckpt="${MC_DIR}/${ckpt}"
  base_ckpt="${BASE_DIR}/${ckpt}"

  echo ""
  echo "=========================================================================="
  echo " EVALUATING ${ep} EPOCH CHECKPOINT (${ckpt})                             "
  echo "=========================================================================="

  # 1. Ko-HellaSwag Evaluation
  echo "[1/3] Ko-HellaSwag Evaluation (${ep} Epochs)..."
  python3 benchmarks/run_ko_hellaswag.py \
    --out_dir "$MC_DIR" \
    --ckpt_path "$mc_ckpt" \
    --device "$DEVICE" \
    --eval_all_norms \
    --output_json "${MC_DIR}/ko_hellaswag_metrics_${ep}ep.json"

  python3 benchmarks/run_ko_hellaswag.py \
    --out_dir "$BASE_DIR" \
    --ckpt_path "$base_ckpt" \
    --device "$DEVICE" \
    --eval_all_norms \
    --output_json "${BASE_DIR}/ko_hellaswag_metrics_${ep}ep.json"

  # 2. Phonetic Slang & Typo Resilience Evaluation
  echo "[2/3] Phonetic Slang Resilience Evaluation (${ep} Epochs)..."
  python3 benchmarks/run_phonetic_slang_eval.py \
    --mc_dir "$MC_DIR" \
    --base_dir "$BASE_DIR" \
    --mc_ckpt "$mc_ckpt" \
    --base_ckpt "$base_ckpt" \
    --dataset_name "Blpeng/nsmc" \
    --split "test" \
    --max_examples 0 \
    --corruption_rate 0.8 \
    --device "$DEVICE" \
    --output_json "./phonetic_slang_resilience_results_${ep}ep.json"

  # 3. Vocabulary Tail Perplexity Evaluation
  echo "[3/3] Vocab Tail Perplexity Evaluation (${ep} Epochs)..."
  python3 benchmarks/run_vocab_tail_perplexity.py \
    --mc_dir "$MC_DIR" \
    --base_dir "$BASE_DIR" \
    --mc_ckpt "$mc_ckpt" \
    --base_ckpt "$base_ckpt" \
    --train_corpus "data/korean_pos_mc/input.txt" \
    --dataset_name "KETI-AIR/kor_hellaswag" \
    --split "validation" \
    --max_examples 500 \
    --device "$DEVICE" \
    --output_json "./vocab_tail_perplexity_results_${ep}ep.json"

done

echo ""
echo "=========================================================================="
echo " All Evaluations on 3, 5, and 10 Epoch Checkpoints Completed Successfully! "
echo "=========================================================================="

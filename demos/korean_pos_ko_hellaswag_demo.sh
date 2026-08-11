#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Korean POS Factorized Tokenizer vs Baseline: Ko-HellaSwag Experiment "
echo "=========================================================================="

MAX_ITERS="${MAX_ITERS:-3000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT:-0.05}"
DROPOUT="${DROPOUT:-0.1}"
DEVICE="${DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"

# 1. Prepare/verify dataset streams using HangulPosFactorizedTokenizer
echo "[Step 1/5] Preparing dataset streams..."
bash data/korean_pos_mc/get_dataset.sh

lanes=(
  korean_pos_mc/script
  korean_pos_mc/choseong
  korean_pos_mc/jungseong
  korean_pos_mc/jongseong
  korean_pos_mc/jung_base1
  korean_pos_mc/jung_base2
  korean_pos_mc/jung_has_w
  korean_pos_mc/jung_has_y
  korean_pos_mc/jung_has_i
  korean_pos_mc/jong_base1
  korean_pos_mc/jong_base2
  korean_pos_mc/jong_base3
  korean_pos_mc/choseong_tense
  korean_pos_mc/choseong_aspirated
  korean_pos_mc/choseong_nasal_liquid
  korean_pos_mc/choseong_place
  korean_pos_mc/jung_height
  korean_pos_mc/jung_backness
  korean_pos_mc/jung_round
  korean_pos_mc/jong_complex
  korean_pos_mc/has_batchim
  korean_pos_mc/syllable_index_mod
  korean_pos_mc/codepoint_mod
  korean_pos_mc/pos
  korean_pos_mc/char
)

# 2. Train Multicontext HangulPosFactorizedTokenizer model
echo ""
echo "[Step 2/5] Training Multicontext model with HangulPosFactorizedTokenizer (25 lanes)..."
echo "Params: max_iters=$MAX_ITERS, structural_loss_weight=$STRUCTURAL_LOSS_WEIGHT, dropout=$DROPOUT"
python3 train.py \
  --dataset korean_pos_mc/char \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${lanes[@]}" \
  --structural_loss_weight "$STRUCTURAL_LOSS_WEIGHT" \
  --max_iters "$MAX_ITERS" \
  --eval_iters "$EVAL_ITERS" \
  --always_save_checkpoint \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --out_dir ./out_mc_korean_pos

# 3. Train Baseline single-context model
echo ""
echo "[Step 3/5] Training Baseline single-context model (character level)..."
echo "Params: max_iters=$MAX_ITERS, dropout=$DROPOUT"
python3 train.py \
  --dataset korean_pos_mc/char \
  --max_iters "$MAX_ITERS" \
  --eval_iters "$EVAL_ITERS" \
  --always_save_checkpoint \
  --dropout "$DROPOUT" \
  --device "$DEVICE" \
  --out_dir ./out_baseline_korean_pos

# 4. Evaluate both models on Ko-HellaSwag
echo ""
echo "[Step 4/5] Running Ko-HellaSwag evaluation across all normalization modes..."

eval_mc_cmd=(python3 benchmarks/run_ko_hellaswag.py --out_dir ./out_mc_korean_pos --device "$DEVICE" --eval_all_norms --output_json ./out_mc_korean_pos/ko_hellaswag_metrics.json)
if [ -n "${MAX_EXAMPLES:-}" ]; then
  eval_mc_cmd+=(--max_examples "$MAX_EXAMPLES")
fi

eval_base_cmd=(python3 benchmarks/run_ko_hellaswag.py --out_dir ./out_baseline_korean_pos --device "$DEVICE" --eval_all_norms --output_json ./out_baseline_korean_pos/ko_hellaswag_metrics.json)
if [ -n "${MAX_EXAMPLES:-}" ]; then
  eval_base_cmd+=(--max_examples "$MAX_EXAMPLES")
fi

echo "Evaluating Multicontext (HangulPosFactorizedTokenizer) model..."
"${eval_mc_cmd[@]}"

echo "Evaluating Baseline (Single Context) model..."
"${eval_base_cmd[@]}"

# 5. Display comparison report
echo ""
echo "=========================================================================="
echo " FINAL EXPERIMENT COMPARISON SUMMARY "
echo "=========================================================================="
python3 -c "
import json
from pathlib import Path

mc_path = Path('./out_mc_korean_pos/ko_hellaswag_metrics.json')
base_path = Path('./out_baseline_korean_pos/ko_hellaswag_metrics.json')

if mc_path.exists() and base_path.exists():
    mc = json.loads(mc_path.read_text())
    base = json.loads(base_path.read_text())

    print(f'Benchmark Dataset : {mc.get(\"dataset_name\")}')
    print(f'Total Evaluated   : {mc.get(\"total\")} examples')
    print('=' * 80)
    print(f'{\"Normalization Mode\":<20} | {\"Baseline Acc\":<15} | {\"Multicontext Acc\":<18} | {\"Delta\":<10}')
    print('-' * 80)

    mc_accs = mc.get('accuracies', {})
    base_accs = base.get('accuracies', {})

    if not mc_accs and 'accuracy' in mc:
        norm_type = mc.get('norm_type', 'length')
        mc_accs = {norm_type: mc.get('accuracy', 0.0)}
        base_accs = {norm_type: base.get('accuracy', 0.0)}

    modes = ['length', 'prior_length', 'unigram_length', 'none', 'prior', 'unigram']
    for mode in modes:
        if mode in mc_accs and mode in base_accs:
            b_acc = base_accs[mode]
            m_acc = mc_accs[mode]
            diff = m_acc - b_acc
            sign = '+' if diff >= 0 else ''
            print(f'{mode:<20} | {b_acc:.4f}          | {m_acc:.4f}             | {sign}{diff:.4f}')
    print('=' * 80)
else:
    print('Error: Metric JSON output files not found.')
"
echo "=========================================================================="

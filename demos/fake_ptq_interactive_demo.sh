#!/bin/bash
# demos/fake_ptq_interactive_demo.sh
#
# Demonstrates the per-tensor configuration and interactive TUI flow for the
# fake PTQ utility. The script trains a compact Shakespeare character model,
# evaluates it, then launches the Textual-based selector so you can choose the
# bit-width for each tensor before quantizing the checkpoint.

set -euo pipefail

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd data/shakespeare_char > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

OUT_DIR="out_fake_ptq_shakespeare"
QUANTIZED_OUT_DIR="${OUT_DIR}_ptq"
LAST_PLAN_BASENAME="last_fake_ptq_quantization.yaml"

echo "=== Step 2: Train a reference model on shakespeare_char ==="
python3 train.py \
  --dataset shakespeare_char \
  --out_dir "$OUT_DIR" \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --block_size 128 \
  --batch_size 64 \
  --max_iters 500 \
  --lr_decay_iters 500 \
  --eval_iters 50 \
  --log_interval 10 \
  --always_save_checkpoint

echo "=== Step 3: Evaluate the baseline checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --init_from resume \
  --eval_only \
  --eval_iters 200 \
  --eval_dataset shakespeare_char

LAST_PLAN_PATH="$OUT_DIR/$LAST_PLAN_BASENAME"
TUI_DEFAULT_ARGS=()

echo "=== Step 4: Apply fake PTQ with interactive refinement ==="
if [ -f "$LAST_PLAN_PATH" ]; then
  echo "Found previous quantization plan at $LAST_PLAN_PATH. It will seed the default column."
  TUI_DEFAULT_ARGS+=(--tui-default-quantization "$LAST_PLAN_PATH")
else
  echo "No saved quantization plan found yet; configure bit-widths manually in the planner."
fi
echo "Tip: When the planner opens, use arrow keys to move, '+'/'-' or digits to edit, and press 'p' for the command menu."
echo "The planner records your selections to $LAST_PLAN_PATH for reuse on subsequent runs."

python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
  --out_dir "$QUANTIZED_OUT_DIR" \
  --num_bits 8 \
  --quantization asymmetric \
  --interactive \
  --min-bits 2 \
  --max-bits 8 \
  --tui-page-size 12 \
  "${TUI_DEFAULT_ARGS[@]}"

if [ -f "$LAST_PLAN_PATH" ]; then
  echo "Latest planner selections saved to $LAST_PLAN_PATH."
fi

echo "=== Step 5: Evaluate the quantized checkpoint ==="
python3 sample.py \
  --out_dir "$QUANTIZED_OUT_DIR" \
  --init_from resume \
  --eval_only \
  --eval_iters 200 \
  --eval_dataset shakespeare_char

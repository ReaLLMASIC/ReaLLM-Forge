#!/bin/bash
# demos/distillation_minipile_comparison.sh
#
# Demonstrates training a baseline student model versus a distilled student
# model on the minipile dataset. The script prepares the dataset, trains a
# larger teacher checkpoint, runs two student trainings (with and without
# distillation), and prints model statistics tables that are saved using the
# RUN_NAME of each run.

set -euo pipefail

DATA_DIR="data/minipile"
PRINT_DIR="print_stats"
TEACHER_OUT="out_distill_teacher"
BASELINE_OUT="out_distill_baseline"
DISTILL_OUT="out_distill_student"

TEACHER_RUN="distill_teacher"
BASELINE_RUN="distill_baseline"
DISTILL_RUN="distill_student"

MAX_ITERS=2000
EVAL_INTERVAL=2000
EVAL_ITERS=200
BATCH_SIZE=48
BLOCK_SIZE=256

mkdir -p "$DATA_DIR" "$PRINT_DIR"

echo "=== Step 1: Prepare the minipile dataset ==="
pushd "$DATA_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

echo "=== Step 2: Train a larger teacher model on minipile ==="
if [ ! -f "$TEACHER_OUT/ckpt.pt" ]; then
  RUN_NAME="$TEACHER_RUN" python3 train.py \
    --dataset minipile \
    --out_dir "$TEACHER_OUT" \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_iters "$MAX_ITERS" \
    --eval_interval "$EVAL_INTERVAL" \
    --eval_iters "$EVAL_ITERS" \
    --eta_variant iteration \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --use_qk_norm \
    --use_qk_norm_scale \
    --compile \
    --compute_model_stats \
    --print_model_stats_table "${PRINT_DIR}/${RUN_NAME}.csv" \
    --tensorboard_run_name "$TEACHER_RUN"
else
  echo "Found existing teacher checkpoint at $TEACHER_OUT/ckpt.pt; skipping training."
fi

echo "=== Step 3: Train a baseline student without distillation ==="
RUN_NAME="$BASELINE_RUN" python3 train.py \
  --dataset minipile \
  --out_dir "$BASELINE_OUT" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size "$BLOCK_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --max_iters "$MAX_ITERS" \
  --eval_interval "$EVAL_INTERVAL" \
  --eval_iters "$EVAL_ITERS" \
  --eta_variant iteration \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_qk_norm \
  --use_qk_norm_scale \
  --compile \
  --compute_model_stats \
  --print_model_stats_table "${PRINT_DIR}/${RUN_NAME}.csv" \
  --tensorboard_run_name "$BASELINE_RUN"

echo "=== Step 4: Train a distilled student using the teacher checkpoint ==="
RUN_NAME="$DISTILL_RUN" python3 train.py \
  --dataset minipile \
  --out_dir "$DISTILL_OUT" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size "$BLOCK_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --max_iters "$MAX_ITERS" \
  --eval_interval "$EVAL_INTERVAL" \
  --eval_iters "$EVAL_ITERS" \
  --eta_variant iteration \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --use_qk_norm \
  --use_qk_norm_scale \
  --compile \
  --compute_model_stats \
  --distillation_teacher_ckpt "$TEACHER_OUT/ckpt.pt" \
  --distillation_loss kl_divergence \
  --distillation_temperature 2.0 \
  --distillation_weight 0.5 \
  --print_model_stats_table "${PRINT_DIR}/${RUN_NAME}.csv" \
  --tensorboard_run_name "$DISTILL_RUN"

echo "=== Step 5: Compare model statistics between baseline and distilled students ==="
python3 view_model_stats.py \
  "${PRINT_DIR}/${BASELINE_RUN}.csv" \
  "${PRINT_DIR}/${DISTILL_RUN}.csv" \
  --stats kurtosis

cat <<EOF

Comparison complete. Inspect TensorBoard runs '$BASELINE_RUN' and '$DISTILL_RUN' for
loss curves, and review the CSV tables in ${PRINT_DIR}/ for detailed model
statistics named after each RUN_NAME.
EOF

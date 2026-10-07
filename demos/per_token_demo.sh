#!/bin/bash

RUN_DIR="out/minipile_metrics_trial_$(date +%Y%m%d_%H%M%S)"

MPLBACKEND=Agg python train.py \
  --dataset minipile \
  --out_dir "$RUN_DIR" \
  --device cuda:0 \
  --dtype bfloat16 \
  --n_layer 2 \
  --n_head 4 \
  --n_kv_group 4 \
  --n_embd 128 \
  --block_size 128 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 0.0003 \
  --warmup_iters 0 \
  --max_iters 100 \
  --eval_interval 50 \
  --eval_iters 2 \
  --log_interval 10 \
  --log_per_token_metrics \
  --export_min_angle_graph_device cuda:0 \
  --export_min_angle_graph_block_size 1024 \
  --only_save_checkpoint_at_end \
  --no-tensorboard_log \
  --no-wandb_log \
  --no-csv_log \
  --no-print_model_info

echo "next open: $RUN_DIR/per_token_metrics/per_token_metrics.html"

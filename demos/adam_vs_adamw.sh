#!/bin/bash
# adam_vs_adamw.sh
# Demonstration comparing Adam and AdamW optimizers on the shakespeare_char dataset.

set -e

pushd data/shakespeare_char
bash get_dataset.sh
popd

# Train a small model with Adam
python3 train.py \
  --dataset shakespeare_char \
  --optimizer adam \
  --out_dir out_adam_demo \
  --max_iters 500 \
  --block_size 128 \
  --learning_rate 3e-4 \
  --compile \
  --no-tensorboard_log \
  --compute_model_stats \
  --print_model_stats_table adam_stats.csv

# Train a similar model with AdamW
python3 train.py \
  --dataset shakespeare_char \
  --optimizer adamw \
  --out_dir out_adamw_demo \
  --max_iters 500 \
  --block_size 128 \
  --learning_rate 3e-4 \
  --compile \
  --no-tensorboard_log \
  --compute_model_stats \
  --print_model_stats_table adamw_stats.csv

# Display the coloured delta between the two runs
python3 view_model_stats.py adam_stats.csv adamw_stats.csv --stats kurtosis

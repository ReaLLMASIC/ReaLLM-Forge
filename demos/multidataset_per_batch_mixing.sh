#!/bin/bash
# demos/multidataset_per_batch_mixing.sh
# Demonstrates scheduled per-batch mixing across multiple datasets.

set -e

# Ensure required corpora are prepared.
bash data/shakespeare_char/get_dataset.sh

pushd data/minipile > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin, val.bin, and meta.pkl already exist for minipile."
fi
popd > /dev/null

# Run a small demonstration training job with per-batch mixing enabled.
python3 train.py \
  --training_mode multidataset \
  --dataset_list shakespeare_char minipile \
  --dataset_mixing_per_batch \
  --dataset_sampling_probs 3 1 \
  --dataset_sampling_probs_final 1 3 \
  --dataset_sampling_probs_transition_method linear \
  --batch_size 8 \
  --block_size 128 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --max_iters 2000 \
  --eval_interval 200 \
  --eval_iters 50 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --optimizer adamw \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --compile \
  --no-tensorboard_log \
  --seed 1337 \
  --out_dir out_multidataset_per_batch_demo

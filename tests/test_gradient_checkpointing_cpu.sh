#!/bin/bash

# head to repo root
cd ../

DATASET="shakespeare_char"
bash "data/${DATASET}/get_dataset.sh"

python3 train.py \
  --out_dir=out_gc \
  --device=cpu \
  --eval_interval=2 \
  --log_interval=1 \
  --block_size=2 \
  --batch_size=2 \
  --n_layer=2 \
  --n_head=2 \
  --n_kv_group=2 \
  --n_embd=16 \
  --max_iters=3 \
  --lr_decay_iters=2 \
  --dropout=0.0 \
  --dataset "$DATASET" \
  --use_gradient_checkpointing

python3 sample.py --device=cpu --out_dir=out_gc


#!/bin/bash
# demos/num_mc.sh

python train.py \
  --training_mode multicontext \
  --dataset sinewave/s1 \
  --multicontext \
  --multicontext_datasets sinewave/s_1 sinewave/s_2 \
  --numerical_multicontext \
  --numerical_mlp_hidden_dim 64 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --block_size 128 \
  --batch_size 32 \
  --max_iters 1000 \
  --eval_interval 100 \
  --out_dir out/numerical_mc_test

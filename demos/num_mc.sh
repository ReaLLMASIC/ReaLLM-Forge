#!/bin/bash
# Demo script for numerical multi-context training with sine wave data.

# python train.py \
#   --training_mode multicontext \
#   --dataset sinewave/s1 \
#   --multicontext \
#   --multicontext_datasets \
#     sinewave/s1 \
#     sinewave/s2 \
#     sinewave/s3 \
#     sinewave/s4 \
#     sinewave/s5 \
#     sinewave/s6 \
#     sinewave/s7 \
#     sinewave/s8 \
#     sinewave/s9 \
#     sinewave/s10 \
#     sinewave/s11 \
#     sinewave/s12 \
#     sinewave/s13 \
#     sinewave/s14 \
#     sinewave/s15 \
#     sinewave/s16 \
#   --numerical_multicontext \
#   --numerical_mlp_hidden_dim 64 \
#   --n_layer 10 \
#   --n_head 6 \
#   --n_embd 384 \
#   --block_size 256 \
#   --batch_size 32 \
#   --max_iters 3000 \
#   --eval_interval 500 \
#   --compile \
#   --out_dir out/numerical_mc_test_2
python train.py \
  --training_mode multicontext \
  --dataset sinewave/s1 \
  --multicontext \
  --multicontext_datasets \
    sinewave/s1 \
    sinewave/s8 \
    sinewave/s11 \
    sinewave/s13 \
    sinewave/s14 \
  --numerical_multicontext \
  --numerical_mlp_hidden_dim 64 \
  --n_layer 10 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --batch_size 32 \
  --max_iters 3000 \
  --eval_interval 500 \
  --compile \
  --out_dir out/numerical_mc_test_2

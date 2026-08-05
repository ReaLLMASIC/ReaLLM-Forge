#!/bin/bash
# recursively goes through the "out" directory and tests different block sizes
python3 demos/ckpt_block_size_eval_plotly.py \
  --block_sizes 256 512 1024 2048 4096 8192 \
  --dtype bfloat16 \
  --dark_mode \
  out

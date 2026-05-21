#!/bin/bash

python latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --split "train[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --num_examples 3 \
  --example_max_new_tokens 64 \
  --route_mode latin_punct_only \
  --byte_fallback \
  --device cuda

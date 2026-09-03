#!/bin/bash

python3 tokenize_and_annotate_sizes.py \
  --in-json filtered_files.json \
  --method tiktoken \
  --tiktoken-encoding gpt2


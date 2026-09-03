#!/bin/bash

python3 plot_tokenization_vs_original.py \
  --json filtered_files.json \
  --mode ratio \
  --method tiktoken \
  --group-by family --color-by family \
  --out ratio_family.png

python3 plot_tokenization_vs_original.py \
  --json filtered_files.json \
  --mode tokenized_kb --method tiktoken \
  --group-by region --color-by region \
  --out tok_kb_region.png

python3 plot_tokenization_vs_original.py \
  --json filtered_files.json \
  --mode ratio --method tiktoken \
  --group-by script --color-by script \
  --out ratio_script.png


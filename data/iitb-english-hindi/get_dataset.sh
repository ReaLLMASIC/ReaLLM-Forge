#!/bin/bash
URL="https://huggingface.co/datasets/cfilt/iitb-english-hindi/tree/main/data"

python utils/get_translation_parquet_dataset.py \
  --url "$URL" \
  --prefix en $'\nEN: ' \
  --prefix hi $'HI: ' \
  --output input.txt


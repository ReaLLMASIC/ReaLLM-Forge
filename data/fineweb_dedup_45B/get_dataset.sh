#!/bin/bash

# Example for: https://huggingface.co/datasets/skymizer/fineweb-edu-dedup-45B/tree/main/data
# Files are named: train-00000-of-00469.parquet to train-00468-of-00469.parquet

BASE_URL="https://huggingface.co/datasets/skymizer/fineweb-edu-dedup-45B/resolve/main/data/train"
START_SHARD=0
END_SHARD=2      # This is the last file index (inclusive)
TOTAL_SHARDS=448   # This is the 'of-XXXXX' number

python3 ./utils/get_parquet_dataset_range.py \
  --url_base "${BASE_URL}" \
  --start_num ${START_SHARD} \
  --stop_num ${END_SHARD} \
  --total_shards ${TOTAL_SHARDS} \
  -i "text" \
  -p "" \
  -o "input.txt" \
  -s

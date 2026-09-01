#!/bin/bash

# FineWeb-Edu sample-10BT
#
# Source: HuggingFaceFW/fineweb-edu (config: sample-10BT)
# Size:  ~9.67M examples, ~10B tokens, 14 parquet files (~28.5GB)
#
# The sample-10BT subset lives under "sample/10BT/" on HuggingFace.
# The HF tree page uses JavaScript rendering so get_parquet_dataset.py
# cannot scrape the download links. We construct direct URLs instead.
#
# prepare.py is NOT used here because it loads the entire input file into
# memory, which OOMs on the ~43GB input.txt. tokenize_chunked.py streams
# line-by-line in 50MB chunks, keeping memory usage under 1GB.

base_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
num_files=14

mkdir -p downloaded_parquets

# Step 1: Download parquet files
for i in $(seq 0 $((num_files - 1))); do
  padded=$(printf "%03d" "$i")
  filename="${padded}_00000.parquet"
  parquet_path="downloaded_parquets/${filename}"

  if [ ! -f "${parquet_path}" ]; then
    echo "Downloading ${filename}..."
    wget -q --show-progress -O "${parquet_path}" "${base_url}/${filename}"
  else
    echo "${filename} already downloaded, skipping."
  fi
done

# Step 2: Extract text from parquets into input.txt
python3 extract_parquets.py downloaded_parquets input.txt

# Step 3: Tokenize (streaming, low memory)
echo "Tokenizing with tiktoken..."
python3 tokenize_chunked.py input.txt
echo "Done."

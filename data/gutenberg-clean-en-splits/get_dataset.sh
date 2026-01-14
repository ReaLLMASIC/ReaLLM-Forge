#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
train_url="https://huggingface.co/datasets/nikolina-p/gutenberg_clean_en_splits/tree/main/data/train"
val_url="https://huggingface.co/datasets/nikolina-p/gutenberg_clean_en_splits/tree/main/data/validation"
test_url="https://huggingface.co/datasets/nikolina-p/gutenberg_clean_en_splits/tree/main/data/test"

python3 ./utils/get_parquet_dataset.py \
  --url "${train_url}" \
  --include_keys "text" \
  --value_prefix $'\n#text:\n' \
  --output_text_file train.txt

python3 ./utils/get_parquet_dataset.py \
  --url "${validation_url}" \
  --include_keys "text" \
  --value_prefix $'\n#text:\n' \
  --output_text_file val.txt

python3 ./utils/get_parquet_dataset.py \
  --url "${test_url}" \
  --include_keys "text" \
  --value_prefix $'\n#text:\n' \
  --output_text_file test.txt

cat train.txt val.txt test.txt > input.txt



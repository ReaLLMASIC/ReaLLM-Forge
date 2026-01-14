#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
test_url="https://huggingface.co/datasets/knkarthick/dialogsum/tree/refs%2Fconvert%2Fparquet/default/test"
train_url="https://huggingface.co/datasets/knkarthick/dialogsum/tree/refs%2Fconvert%2Fparquet/default/train"
validation_url="https://huggingface.co/datasets/knkarthick/dialogsum/tree/refs%2Fconvert%2Fparquet/default/validation"

declare -A url_array
url_array["train"]="$train_url"
url_array["validation"]="$validation_url"
url_array["test"]="$test_url"

for split_name in "${!url_array[@]}"; do
  url="${url_array[$split_name]}"
  python3 ./utils/get_parquet_dataset.py \
    --url "$url" \
    --include_keys "dialogue" "summary" \
    --value_prefix $'\n#U: Please summarize the following:\n' $'\n#B:\n' \
    --output_text_file "${split_name}.txt"
done

rm input.txt

for split_name in "${!url_array[@]}"; do
  text_file="${split_name}.txt"
  cat "$text_file" >> input.txt
done


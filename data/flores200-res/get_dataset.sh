#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.
# 6. For CSV files with BOM headers, pass "--input_encoding utf-8-sig" to the helper script.
# 7. For CSV cells that contain multi-line text, use "--split_multiline_values" to emit one line per entry or
#    "--newline_replacement" to substitute newline characters with custom text.

# Run the Python script with the specified arguments

lang_array=(
  "text_eng_Latn"
  "text_jpn_Jpan"
  "text_kor_Hang"
  "text_zho_Hans"
)

# Add url with dataset here:
url="https://huggingface.co/datasets/muhammadravi251001/restructured-flores200/tree/main/data"

for lang in "${lang_array[@]}"; do
  python3 ./utils/get_parquet_dataset.py \
    --url "${url}" \
    --include_keys "$lang" \
    --value_prefix $'\n' \
    --output_text_file "$lang".txt

done

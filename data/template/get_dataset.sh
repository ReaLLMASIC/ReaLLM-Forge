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

# Add url with dataset here:
url="INSERT_URL_WITH_FILES"

# uncomment and fill in if url has json datasets
# Note: the $'\n' syntax allows for special characters like \n
# python3 ./utils/get_json_dataset.py \
#   --url "${url}" \
#   --include_keys "instruction" "response" \
#   --value_prefix $'#U:\n' $'#B:\n'

# uncomment and fill in if url has parquet datasets
# python3 ./utils/get_parquet_dataset.py \
#   --url "${url}" \
#   --include_keys "instruction" "response" \
#   --value_prefix $'#U:\n' $'#B:\n'
#
# uncomment and fill in if url has csv datasets
# python3 ./utils/get_csv_dataset.py \
#   --url "${url}" \
#   --include_keys "instruction" "response" \
#   --value_prefixes $'#U:\n' $'#B:\n' \
#   --split_by_prefix
#   # Uncomment one of the following lines to control newline handling:
#   # --split_multiline_values \
#   # --newline_replacement "\\n" \
#

# uncomment for direct to json
# python utils/get_parquet_to_json.py \
#   --url "${url}" \
#   --range_start 0 \
#   --range_end 1 \
#   --include_keys transcription \
#   --output_json output_json.json

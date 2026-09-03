#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

python3 "$script_dir"/utils/zh_to_ipa.py zh_cn.txt ipa_text_zho_Hans.txt --input_type text --no-wrapper --stats_json zh_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_zho_Hans.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt
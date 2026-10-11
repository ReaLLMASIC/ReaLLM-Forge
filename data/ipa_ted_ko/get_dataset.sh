#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

python3 "$script_dir"/utils/ko_en_to_ipa.py ko.txt --text_input --text_output ipa_text_kor_Hang.txt --stats_json ko_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_kor_Hang.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt
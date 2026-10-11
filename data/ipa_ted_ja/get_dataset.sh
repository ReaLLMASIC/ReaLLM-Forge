#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

python3 "$script_dir"/utils/ja2ipa.py ja.txt ipa_text_jpn_Jpan.txt --text_output --use_spacy --text_no_sentence --stats_json ja_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_jpn_Jpan.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt
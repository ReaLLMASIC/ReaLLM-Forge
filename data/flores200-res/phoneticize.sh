#!/bin/bash


python3 utils/en2ipa.py text_eng_Latn.txt --mode text --output_file ipa_text_eng_Latn.txt --no-wrapper --stats_json eng_stats.json
python3 utils/ja2ipa.py text_jpn_Jpan.txt ipa_text_jpn_Jpan.txt --text_output --use_spacy --text_no_sentence --stats_json ja_stats.json
python3 utils/ko_en_to_ipa.py text_kor_Hang.txt --text_input --text_output ipa_text_kor_Hang.txt --stats_json ko_stats.json
python3 utils/zh_to_ipa.py text_zho_Hans.txt ipa_text_zho_Hans.txt --input_type text --no-wrapper --stats_json zh_stats.json
# python3 utils/espeak2ipa.py text_shn_Mymr.txt --mode text --output_file ipa_text_shan.txt --no-wrapper --stats_json shan_stats.json --lang shan


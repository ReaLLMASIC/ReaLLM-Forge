#!/bin/bash


# python3 utils/en2ipa.py text_eng_Latn.txt --mode text --output_file ipa_text_eng_Latn.txt --no-wrapper --stats_json eng_stats.json
# python3 utils/ja2ipa.py text_jpn_Jpan.txt ipa_text_jpn_Jpan.txt --text_output --use_spacy --text_no_sentence --stats_json ja_stats.json
# python3 utils/ko_en_to_ipa.py text_kor_Hang.txt --text_input --text_output ipa_text_kor_Hang.txt --stats_json ko_stats.json
# python3 utils/zh_to_ipa.py text_zho_Hans.txt ipa_text_zho_Hans.txt --input_type text --no-wrapper --stats_json zh_stats.json

lang_array=(
  "text_vie_Latn:vi"
  "text_ind_Latn:id"
  "text_swh_Latn:sw"
  "text_ell_Grek:el"
  "text_fra_Latn:fr"
  "text_yue_Hant:yue"
)

for lang in "${lang_array[@]}"; do
  text_file="${lang%%:*}"
  two_letter_code="${lang##*:}"
  echo "${text_file}; ${two_letter_code}"
  if [ ! -f "ipa_${text_file}.txt" ]; then
    python3 utils/espeak2ipa.py "$text_file".txt --mode text --output_file ipa_"$text_file".txt --no-wrapper --stats_json stats_"$text_file".json --lang "$two_letter_code" --text_no_sentence
  fi
done

#!/bin/bash
# data/template/utils/en2ipa_wrapper.sh

input_file="${1:-input.txt}"
snippet_size="${2:-300000}"
snippet_file="snippet_${snippet_size}.txt"

head -n "$snippet_size" "$input_file" > "$snippet_file"
python3 utils/en2ipa.py "$snippet_file" --mode text --multithread --outputfile input_"$snippet_size".txt

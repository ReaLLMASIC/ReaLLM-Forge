# !/bin/bash

set -xe

# Install python dependencies for Hugging face
pip install -U "huggingface_hub[cli]"

# Authentication with Hugging Face
# Replace with your hugging face tokens
##### You can find and create your own tokens here: https://huggingface.co/settings/tokens ######
##### "Token Type" of "Read" is recommended. ########
if [[ -f ~/.cache/huggingface/token && -s ~/.cache/huggingface/token ]]; then
  export HF_TOKEN=$(cat ~/.cache/huggingface/token)
else
  echo "Consider running 'python3 ./utils/save_hf_token.py' to automate finding HF_TOKEN"
  read -s -p "To continue, please enter your Hugging Face token: " HF_TOKEN
  echo "" # Add a newline for better readability
fi

# Authenticate with hugging face
echo "Authenticating with Hugging Face..."
huggingface-cli login --token "${HF_TOKEN}"

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

url="https://huggingface.co/datasets/xinyixuu/zh_snac"
out_dir="out_ipa"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

pushd "${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh_ws_ipa.json" "${url}/resolve/main/zh_ws_ipa.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "zh_ipa.json" "${url}/resolve/main/zh_ipa.json?download=true" || true

echo "json files downloaded and saved to out_ipa."
popd

output_ipa_txt="zh_ipa.txt"
for jsonfile in "$out_dir"/*.json; do
    # Check if the .json file exists (handles the case where no .json files are present)
    if [ -f "$jsonfile" ]; then
        echo "Processing $jsonfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${jsonfile%.json}")
        python3 "$script_dir"/utils/extract_json_values.py "$jsonfile" "sentence_ipa" "$output_ipa_txt"
    fi
done
echo "IPA extraction finished."

#TODO(gkielian): see if we can fix the parsing of rows instead of deleting
# Remove lines which were not correclty processed (and start with numberic hash)
wc -l "$output_ipa_txt"
sed -i "/^[0-9].*/g" "$output_ipa_txt"
wc -l "$output_ipa_txt"

# Tokenization step to create train.bin and val.bin files.
#python3 "$script_dir"/prepare.py -t "$output_ipa_txt" --method char
python3 "$script_dir"/prepare.py -t "$output_ipa_txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt

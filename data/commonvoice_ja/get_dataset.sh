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

url="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
out_dir="transcription"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

# Download transcription files under "transcription" directory.
pushd "${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.tsv" "${url}/resolve/main/transcript/ja/dev.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.tsv" "${url}/resolve/main/transcript/ja/validated.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.tsv" "${url}/resolve/main/transcript/ja/other.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.tsv" "${url}/resolve/main/transcript/ja/test.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.tsv" "${url}/resolve/main/transcript/ja/train.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.tsv" "${url}/resolve/main/transcript/ja/validated.tsv?download=true" || true

echo "transcripts downloaded and saved to transcription."
popd

# Run program to convert tsv into json format.
output_file="ja_transcription.json"
for tsvfile in "$out_dir"/*.tsv; do
    # Check if the .tsv file exists (handles the case where no .tsv files are present)
    if [ -f "$tsvfile" ]; then
        echo "Processing $tsvfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${tsvfile%.tsv}")
        python3 "$script_dir"/utils/tsv_to_json_cv_pandas.py "$tsvfile" "$output_file"
    fi
done

echo "All .tsv files have been processed."

# # Run program to convert sentences into IPA format.
output_json_with_ipa="ja_ipa.json"
echo "Converting sentences to IPA..."
python3 "$script_dir"/utils/ja2ipa.py  -j "$output_file" "$output_json_with_ipa" --use_mecab
echo "IPA conversion finished."

output_ipa_txt="ja_ipa.txt"

# Download kokoro dataset from huggingface
ja_dataset="kokoro_transcription"
ja_url="https://huggingface.co/datasets/xinyixuu/ja_snac"
if [[ ! -d "${ja_dataset}" ]]; then
  mkdir -p "${ja_dataset}"
fi

# Download transcription files under "transcription" directory.
pushd "${ja_dataset}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "tiny.json" "${ja_url}/resolve/main/json_outs_ja/tiny.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "final.json" "${ja_url}/resolve/main/json_outs_ja/final.json?download=true" || true
echo "kokoro transcripts downloaded and saved to kokoro_transcription."
popd
python3 "$script_dir"/utils/extract_json_values.py "$output_json_with_ipa" "spaced_ipa" "$output_ipa_txt"

for jsonfile in "$ja_dataset"/*.json; do
    # Check if the .json file exists (handles the case where no .json files are present)
    if [ -f "$jsonfile" ]; then
        echo "Processing $jsonfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${jsonfile%.json}")
        python3 "$script_dir"/utils/extract_json_values.py "$jsonfile" "spaced_ipa" "$output_ipa_txt"
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

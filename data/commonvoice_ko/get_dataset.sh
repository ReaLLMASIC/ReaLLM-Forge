# !/bin/bash

# Show lines before execution and exit on errors
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
hf auth login --token "${HF_TOKEN}"

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

url="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
out_dir="transcription"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

# Download transcription files under "transcription" directory.
pushd "${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.tsv" "${url}/resolve/main/transcript/ko/dev.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.tsv" "${url}/resolve/main/transcript/ko/validated.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.tsv" "${url}/resolve/main/transcript/ko/other.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.tsv" "${url}/resolve/main/transcript/ko/test.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.tsv" "${url}/resolve/main/transcript/ko/train.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.tsv" "${url}/resolve/main/transcript/ko/validated.tsv?download=true" || true

echo "transcripts downloaded and saved to transcription."
popd

# Run program to convert tsv into json format.
output_file="ko_transcription.json"
for tsvfile in "$out_dir"/*.tsv; do
    # Check if the .tsv file exists (handles the case where no .tsv files are present)
    if [ -f "$tsvfile" ]; then
        echo "Processing $tsvfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${tsvfile%.tsv}")
        python3 "$script_dir"/utils/tsv_to_json_cv.py "$tsvfile" "$output_file"
    fi
done

echo "All .tsv files have been processed."

# Run program to convert sentences into IPA format.
echo "Converting sentences to IPA..."
python3 "$script_dir"/utils/ko_en_to_ipa.py "$output_file" --input_json_key "sentence" --output_json_key "sentence_ipa"

output_ipa="ko_ipa.txt"
echo "export IPA to txt file"

# Download kokoro dataset from huggingface
ko_dataset="korean_speech_transcription"
ko_url="https://huggingface.co/datasets/xinyixuu/ko_snac"
if [[ ! -d "${ko_dataset}" ]]; then
  mkdir -p "${ko_dataset}"
fi

# Download transcription files under "transcription" directory.
pushd "${ko_dataset}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac.json" "${ko_url}/resolve/main/json_dir/ko_snac.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_1.json" "${ko_url}/resolve/main/json_dir/ko_snac_1.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_3.json" "${ko_url}/resolve/main/json_dir/ko_snac_3.json?download=true" || true
echo "Korean_Speech_Dataset transcripts downloaded and saved to korean_speech_transcription."
popd

python3 "$script_dir"/utils/extract_json_values.py "$output_file" "sentence_ipa" "$output_ipa"

for jsonfile in "$ko_dataset"/*.json; do
    # Check if the .json file exists (handles the case where no .json files are present)
    if [ -f "$jsonfile" ]; then
        echo "Processing $jsonfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${jsonfile%.json}")
        python3 "$script_dir"/utils/extract_json_values.py "$jsonfile" "ipa" "$output_ipa"
    fi
done

echo "IPA conversion finished."

# Tokenization step to create train.bin and val.bin files.
#python3 "$script_dir"/prepare.py -t "$output_ipa" --method char
python3 "$script_dir"/prepare.py -t "$output_ipa" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt

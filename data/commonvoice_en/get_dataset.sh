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
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.tsv" "${url}/resolve/main/transcript/en/dev.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.tsv" "${url}/resolve/main/transcript/en/validated.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.tsv" "${url}/resolve/main/transcript/en/other.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.tsv" "${url}/resolve/main/transcript/en/test.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.tsv" "${url}/resolve/main/transcript/en/train.tsv?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.tsv" "${url}/resolve/main/transcript/en/validated.tsv?download=true" || true

echo "transcripts downloaded and saved to transcription."
popd

# Run program to convert tsv into json format.
output_file="en_transcription.json"
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
python3 "$script_dir"/utils/en2ipa.py "$output_file" --input_json_key "sentence" --output_json_key "sentence_ipa"

output_ipa="en_ipa.txt"
echo "export IPA to txt file"

python3 "$script_dir"/utils/extract_json_values.py "$output_file" "sentence_ipa" "$output_ipa"

echo "IPA conversion finished."

# Tokenization step to create train.bin and val.bin files.
#python3 "$script_dir"/prepare.py -t "$output_ipa" --method char
python3 "$script_dir"/prepare.py -t "$output_ipa" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt

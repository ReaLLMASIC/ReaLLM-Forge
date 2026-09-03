#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Running snac to audio conversion..."

sample_outs="out_test"
# Change the name of the directory if applicable
pushd ../../
directory="out_test"
echo "$directory"

for item in $directory/*; do
    if [ -d "$item" ]; then
        dirname=$(basename "$item")
        new_dirname="new_$directory"
        new_dirpath="$directory/$new_dirname"
        mv "$item" "$new_dirpath"
        echo "Renamed directory $item to $new_dirpath"
    fi
done

# Run the sample file to get the output file
python3 sample.py --out_dir=${new_dirpath} --start $"#U:hello\n#B:" --sample_file text_snac.txt
popd

# Run program to get the snac file
python3 extract_snac.py ../../text_snac.txt out_snac.txt

# Remove frames which do not have 7 entries
python3 format.py clean out_snac.txt output.txt

# Run snac decoding to get the audio
python3 snac_converter.py decode output.txt out_audio.mp3 --input_format text


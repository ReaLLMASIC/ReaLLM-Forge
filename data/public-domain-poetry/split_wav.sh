#!/bin/bash

# Define the input file name variable
INPUT_FILE="input.wav"

# Check if the file actually exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# 1. Get the total duration in seconds using ffprobe
TOTAL_DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nocut=1 -of csv=p=0 "$INPUT_FILE")

# 2. Use bc to calculate the exact 1/10th duration (with 4 decimal precision)
SEGMENT_TIME=$(echo "scale=4; $TOTAL_DURATION / 10" | bc -l)

# Extract filename without extension for clean output naming
BASE_NAME="${INPUT_FILE%.*}"

echo "Total Duration: $TOTAL_DURATION seconds"
echo "Splitting into 10 pieces of ${SEGMENT_TIME}s each..."

# 3. Split the file using ffmpeg without re-encoding
ffmpeg -v warning -i "$INPUT_FILE" -f segment -segment_time "$SEGMENT_TIME" \
    -c copy "${BASE_NAME}_part_%02d.wav"

echo "Done! Check your folder for ${BASE_NAME}_part_00.wav to ${BASE_NAME}_part_09.wav"


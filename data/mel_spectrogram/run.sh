#!/bin/bash

set -x

# point this at an flac or wav file

bash helpers/convert_to_wav.sh "${1}"

python audio_to_token_mel.py "${1%%.flac}.wav" --force \
  --preset max \
  --samples-per-second 48000 \
  --fmin 10 \
  --fmax 20000 \
  --columns-per-timestep 384 \
  --states-per-column 64 \
  --timestep-ms 15 \
  --win-ms 60 \
  --n-fft 8192 \
  --top-db 96 \
  --reference-mode file_percentile \
  --output-format both \
  --output-dir mel_out


python token_mel_to_audio.py \
  "mel_out/${1%%.flac}.max.mel.csv" \
  -o out.wav \
  --griffin-lim-iters 128 \
  --griffin-lim-chunk-frames 0 \
  --peak 1.0 \
  --force ; mpv out.wav


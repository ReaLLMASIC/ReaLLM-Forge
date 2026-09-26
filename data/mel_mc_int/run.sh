#!/usr/bin/env bash
# Build a mel integer multicontext dataset from one audio file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEL_DIR="${SCRIPT_DIR}/../mel_spectrogram"
INPUT="${1:?Usage: bash data/mel_mc_int/run.sh INPUT_AUDIO [OUTPUT_ROOT]}"
OUTPUT_ROOT="${2:-mel_mc_int}"
WORK_DIR="${SCRIPT_DIR}/mel_out"

mkdir -p "${WORK_DIR}"

python3 "${MEL_DIR}/audio_to_token_mel.py" "${INPUT}" --force \
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
  --output-dir "${WORK_DIR}"

stem="$(basename "${INPUT}")"; stem="${stem%.*}"
MEL_CSV="${WORK_DIR}/${stem}.max.mel.csv"
python3 "${SCRIPT_DIR}/mel_mc_int_tools.py" prepare "${MEL_CSV}" \
  --output_root "${OUTPUT_ROOT}" \
  --states_per_column 64

echo "Dataset manifest: data/${OUTPUT_ROOT}/manifest.json"

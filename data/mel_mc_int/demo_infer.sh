#!/usr/bin/env bash
# Encode an audio prefix, run multicontext sampling, reconstruct continuation audio, and create a viewer.
set -euo pipefail

usage() { echo "Usage: bash data/mel_mc_int/demo_infer.sh OUT_DIR INPUT_AUDIO CUTOFF_SECONDS [MAX_NEW_TOKENS]"; }
[[ $# -ge 3 ]] || { usage >&2; exit 1; }
RUN_OUT_DIR="$1"; INPUT="$2"; CUTOFF="$3"; MAX_NEW="${4:-200}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MEL_DIR="${REPO_ROOT}/data/mel_spectrogram"
OUT="${SCRIPT_DIR}/demo_out/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"

python3 "${MEL_DIR}/audio_to_token_mel.py" "${INPUT}" --force \
  --preset max --samples-per-second 48000 --fmin 10 --fmax 20000 \
  --columns-per-timestep 384 --states-per-column 64 --timestep-ms 15 \
  --win-ms 60 --n-fft 8192 --top-db 96 --reference-mode file_percentile \
  --output-format csv --output-dir "${OUT}"
stem="$(basename "${INPUT}")"; stem="${stem%.*}"
REF_CSV="${OUT}/${stem}.max.mel.csv"
python3 "${SCRIPT_DIR}/mel_mc_int_tools.py" cut-prompt "${REF_CSV}" --cutoff_s "${CUTOFF}" --output_dir "${OUT}/prompt"

mapfile -t DATASETS < <(python3 - <<'PY'
import json
m=json.load(open('data/mel_mc_int/manifest.json'))
print('\n'.join(m['multicontext_datasets']))
PY
)
python3 sample.py --init_from resume --out_dir "${RUN_OUT_DIR}" \
  --multicontext --multicontext_datasets "${DATASETS[@]}" \
  --multicontext_csv_input "${OUT}/prompt/prompt.csv" \
  --multicontext_csv_output_file "${OUT}/generated.csv" \
  --max_new_tokens "${MAX_NEW}" --num_samples 1

python3 "${SCRIPT_DIR}/mel_mc_int_tools.py" wrap-csv "${OUT}/generated.csv" --reference_mel_csv "${REF_CSV}" --output_csv "${OUT}/generated.mel.csv"
python3 "${MEL_DIR}/token_mel_to_audio.py" "${OUT}/generated.mel.csv" --output "${OUT}/generated.wav" --force
python3 "${SCRIPT_DIR}/make_viewer.py" --output_dir "${OUT}" --input_audio "${INPUT}" --cutoff_s "${CUTOFF}"
echo "Viewer: ${OUT}/index.html"

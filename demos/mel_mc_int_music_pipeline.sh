#!/usr/bin/env bash
# End-to-end mel multicontext music demo:
# 1) encode every audio file in a folder to mel CSV with the mel_spectrogram/run.sh defaults
# 2) concatenate the mel-state CSV rows into one training CSV
# 3) split the concatenated CSV into one integer multicontext dataset per mel band
# 4) train a regular multicontext model
# 5) run continuation inference from an input audio prefix and build the HTML viewer

set -euo pipefail
set -x

usage() {
  cat <<'USAGE'
Usage: bash demos/mel_mc_int_music_pipeline.sh MUSIC_DIR [PROMPT_AUDIO] [CUTOFF_SECONDS]

Environment overrides:
  MEL_MC_OUTPUT_ROOT      Dataset folder under data/ (default: mel_mc_int_music)
  MEL_MC_WORK_DIR         Intermediate outputs (default: data/mel_mc_int/music_pipeline_out)
  MEL_MC_OUT_DIR          Training checkpoint dir (default: out/mel_mc_int_music)
  MEL_MC_MAX_ITERS        Training iterations (default: 1000)
  MEL_MC_DEVICE           Device for train/sample/audio tools (default: cuda:0)
  MEL_MC_DTYPE            Training/sample dtype (default: bfloat16)
  MEL_MC_MAX_NEW_TOKENS   Inference frames to sample (default: 200)
  MEL_MC_SKIP_ENCODE      Reuse existing per-file CSVs in WORK_DIR/encoded (default: 0)
  MEL_MC_SKIP_TRAIN       Prepare data and run inference with existing OUT_DIR/ckpt.pt (default: 0)
  MEL_MC_TENSORBOARD      Enable TensorBoard logging during training (default: 0)
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MUSIC_DIR="$1"
PROMPT_AUDIO="${2:-}"
CUTOFF_SECONDS="${3:-10.0}"
OUTPUT_ROOT="${MEL_MC_OUTPUT_ROOT:-mel_mc_int_music}"
WORK_DIR="${MEL_MC_WORK_DIR:-data/mel_mc_int/music_pipeline_out}"
ENCODED_DIR="${WORK_DIR}/encoded"
CONCAT_CSV="${WORK_DIR}/all_music.max.mel.csv"
CONCAT_MANIFEST="${WORK_DIR}/concat_manifest.json"
OUT_DIR="${MEL_MC_OUT_DIR:-out/mel_mc_int_music}"
MAX_ITERS="${MEL_MC_MAX_ITERS:-10000}"
DEVICE="${MEL_MC_DEVICE:-cuda:0}"
DTYPE="${MEL_MC_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MEL_MC_MAX_NEW_TOKENS:-380}"
COMPILE_FLAG="${MEL_MC_COMPILE:-1}"
SKIP_ENCODE="${MEL_MC_SKIP_ENCODE:-1}"
SKIP_TRAIN="${MEL_MC_SKIP_TRAIN:-0}"
TENSORBOARD_FLAG="${MEL_MC_TENSORBOARD:-0}"
MEL_DIR="data/mel_spectrogram"
TOOLS="data/mel_mc_int/mel_mc_int_tools.py"

if [[ ! -d "$MUSIC_DIR" ]]; then
  echo "Music directory does not exist: $MUSIC_DIR" >&2
  exit 1
fi

mkdir -p "$ENCODED_DIR" "$WORK_DIR" "$OUT_DIR"

mapfile -d '' AUDIO_FILES < <(
  find "$MUSIC_DIR" -maxdepth 1 -type f \
    \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.opus' \) \
    -print0 | sort -z
)

if (( ${#AUDIO_FILES[@]} == 0 )); then
  echo "No supported audio files found in $MUSIC_DIR" >&2
  exit 1
fi

if [[ -z "$PROMPT_AUDIO" ]]; then
  PROMPT_AUDIO="${AUDIO_FILES[0]}"
fi

if [[ "$SKIP_ENCODE" != "1" ]]; then
  for audio in "${AUDIO_FILES[@]}"; do
    echo "Encoding $audio"
    python3 "${MEL_DIR}/audio_to_token_mel.py" "$audio" --force \
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
      --output-format csv \
      --output-dir "$ENCODED_DIR" \
      --device "$DEVICE"
  done
fi

mapfile -d '' MEL_CSVS < <(find "$ENCODED_DIR" -maxdepth 1 -type f -name '*.max.mel.csv' -print0 | sort -z)
if (( ${#MEL_CSVS[@]} == 0 )); then
  echo "No encoded mel CSVs found in $ENCODED_DIR" >&2
  exit 1
fi

CSV_INPUT_LIST="$WORK_DIR/encoded_csvs.txt"
printf '%s\n' "${MEL_CSVS[@]}" > "$CSV_INPUT_LIST"
python3 "$TOOLS" concat-csv --input_list "$CSV_INPUT_LIST" \
  --output_csv "$CONCAT_CSV" \
  --manifest_json "$CONCAT_MANIFEST"

python3 "$TOOLS" prepare "$CONCAT_CSV" \
  --output_root "$OUTPUT_ROOT" \
  --states_per_column 64

mapfile -t DATASETS < <(python3 - <<PY
import json
from pathlib import Path
manifest = json.loads(Path('data/$OUTPUT_ROOT/manifest.json').read_text())
for dataset in manifest['multicontext_datasets']:
    print(dataset)
PY
)

if [[ "$COMPILE_FLAG" == "1" || "$COMPILE_FLAG" == "true" || "$COMPILE_FLAG" == "True" ]]; then
  TRAIN_COMPILE_ARG="--compile"
else
  TRAIN_COMPILE_ARG="--no-compile"
fi
if [[ "$TENSORBOARD_FLAG" == "1" || "$TENSORBOARD_FLAG" == "true" || "$TENSORBOARD_FLAG" == "True" ]]; then
  TRAIN_TENSORBOARD_ARG="--tensorboard_log"
else
  TRAIN_TENSORBOARD_ARG="--no-tensorboard_log"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 train.py \
    --training_mode multicontext \
    --dataset "${DATASETS[0]}" \
    --multicontext \
    --multicontext_datasets "${DATASETS[@]}" \
    --n_layer "${MEL_MC_N_LAYER:-12}" \
    --n_embd "${MEL_MC_N_EMBD:-500}" \
    --mlp_size 1536 \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --block_size "${MEL_MC_BLOCK_SIZE:-380}" \
    --batch_size "${MEL_MC_BATCH_SIZE:-12}" \
    --attention_variant infinite \
    --use_concat_heads \
    --no-use_abs_pos_embeddings \
    --use_rotary_embeddings \
    --optimizer muon \
    --weight_decay 0.0 \
    --use_qk_norm \
    --use_qk_norm_scale \
    --max_iters "$MAX_ITERS" \
    --eval_interval "${MEL_MC_EVAL_INTERVAL:-100}" \
    --eval_iters "${MEL_MC_EVAL_ITERS:-10}" \
    --learning_rate "${MEL_MC_LR:-1e-3}" \
    --dropout "${MEL_MC_DROPOUT:-0.0}" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "$TRAIN_COMPILE_ARG" \
    "$TRAIN_TENSORBOARD_ARG" \
    --out_dir "$OUT_DIR"
fi

bash data/mel_mc_int/demo_infer.sh "$OUT_DIR" "$PROMPT_AUDIO" "$CUTOFF_SECONDS" "$MAX_NEW_TOKENS"

#!/bin/bash
# char_bpe_flores_validation_bpb_demo.sh
#
# Demonstrates how to compare FLORES-200 char-BPE tokenizers by training the
# same small model on each language/vocab-size pair and reporting validation
# loss plus bits per byte (BPB). BPB normalizes token-level loss by the number
# of UTF-8 bytes represented per validation token:
#
#   bits_per_byte = validation_loss_nats / (ln(2) * val_bytes_per_token)
#
# The default settings are intentionally small so the script is practical as a
# demo. Override LANGUAGES, VOCAB_SIZES, MAX_ITERS, etc. for larger sweeps.

set -euo pipefail

BASE_DIR="data/char_bpe_exploration"
DATASET_ROOT="data/char_bpe_exploration_train"
OUT_ROOT="out/char_bpe_flores_validation_bpb"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
REPORT_HTML="${OUT_ROOT}/report.html"

LANGUAGES=${LANGUAGES:-"kiswahili bahasa_indonesian korean korean_nfd english chinese japanese arabic spanish german russian thai filipino hindi finnish italian"}
VOCAB_SIZES=${VOCAB_SIZES:-"320 384 512 640 768 1024 1280 1536 2048 3072 4096"}
MAX_ITERS=${MAX_ITERS:-3000}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
EVAL_ITERS=${EVAL_ITERS:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
BLOCK_SIZE=${BLOCK_SIZE:-256}
N_LAYER=${N_LAYER:-6}
N_HEAD=${N_HEAD:-3}
N_EMBD=${N_EMBD:-128}
DEVICE=${DEVICE:-"cuda:0"}
DTYPE=${DTYPE:-"bfloat16"}
PERCENTAGE_TRAIN=${PERCENTAGE_TRAIN:-"0.9"}

mkdir -p "${DATASET_ROOT}" "${OUT_ROOT}"

VOCAB_CSV=$(python3 - <<'PY' "${VOCAB_SIZES}"
import sys
print(",".join(sys.argv[1].split()))
PY
)

echo "=== Step 1: Build FLORES text files and char-BPE train/val bins ==="
python3 "${BASE_DIR}/scripts/run_char_bpe_exploration.py" \
  --vocab-sizes "${VOCAB_CSV}" \
  --percentage-train "${PERCENTAGE_TRAIN}"

echo "language,vocab_size,dataset,validation_loss,bits_per_byte,out_dir" > "${SUMMARY_CSV}"

for language in ${LANGUAGES}; do
  for vocab_size in ${VOCAB_SIZES}; do
    run_dir="${BASE_DIR}/runs/${language}/vocab_${vocab_size}"
    dataset="char_bpe_exploration_train/${language}_vocab_${vocab_size}"
    dataset_dir="data/${dataset}"
    out_dir="${OUT_ROOT}/${language}/vocab_${vocab_size}"

    if [ ! -f "${run_dir}/train.bin" ] || [ ! -f "${run_dir}/val.bin" ] || [ ! -f "${run_dir}/meta.pkl" ]; then
      echo "Missing tokenized artifacts in ${run_dir}" >&2
      exit 1
    fi

    mkdir -p "${dataset_dir}" "${out_dir}"
    cp "${run_dir}/train.bin" "${dataset_dir}/train.bin"
    cp "${run_dir}/val.bin" "${dataset_dir}/val.bin"
    cp "${run_dir}/meta.pkl" "${dataset_dir}/meta.pkl"

    echo "=== Step 2: Train ${language} vocab=${vocab_size} ==="
    python3 train.py \
      --dataset "${dataset}" \
      --out_dir "${out_dir}" \
      --block_size "${BLOCK_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --n_layer "${N_LAYER}" \
      --n_head "${N_HEAD}" \
      --n_embd "${N_EMBD}" \
      --use_rotary_embeddings \
      --no-use_abs_pos_embeddings \
      --attention_variant infinite \
      --use_concat_heads \
      --n_qk_head_dim 120 \
      --n_v_head_dim 120 \
      --use_qk_norm \
      --use_qk_norm_scale \
      --activation_variant squared_relu \
      --use_peri_ln \
      --max_iters "${MAX_ITERS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --eval_iters "${EVAL_ITERS}" \
      --learning_rate 0.001 \
      --weight_decay 0.0 \
      --optimizer muon \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --compile

    echo "=== Step 3: Compute BPB for ${language} vocab=${vocab_size} ==="
    python3 - <<'PY' "${language}" "${vocab_size}" "${dataset}" "${run_dir}/metrics.json" "${out_dir}" "${SUMMARY_CSV}"
import csv
import json
import math
import sys
from pathlib import Path

language, vocab_size, dataset, metrics_path, out_dir, summary_csv = sys.argv[1:]
best_path = Path(out_dir) / "best_val_loss_and_iter.txt"
if not best_path.exists():
    raise SystemExit(f"Missing validation-loss file: {best_path}")
validation_loss = float(best_path.read_text(encoding="utf-8").splitlines()[0].split(",")[0])
metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
val_bytes_per_token = float(metrics["val_bytes_per_token"])
if val_bytes_per_token <= 0:
    raise SystemExit(f"val_bytes_per_token must be > 0 in {metrics_path}")
bits_per_byte = validation_loss / (math.log(2) * val_bytes_per_token)
row = {
    "language": language,
    "vocab_size": vocab_size,
    "dataset": dataset,
    "validation_loss": f"{validation_loss:.6f}",
    "bits_per_byte": f"{bits_per_byte:.6f}",
    "out_dir": out_dir,
}
with Path(summary_csv).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writerow(row)
print(json.dumps(row, indent=2))
PY
  done
done

echo "=== Step 4: Render Plotly HTML report ==="
python3 data/char_bpe_exploration/scripts/render_bpb_report.py \
  --training-summary "${SUMMARY_CSV}" \
  --tokenization-summary "${BASE_DIR}/results/summary.csv" \
  --output "${REPORT_HTML}"

echo "Comparison complete: ${SUMMARY_CSV}"
echo "Plotly report: ${REPORT_HTML}"

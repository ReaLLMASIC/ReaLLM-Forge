#!/usr/bin/env bash
# Dataset feature demo: CSV with 256 explicit pixel columns px_000..px_255 for each 16x16 grayscale frame.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/10_wide_csv_pixels}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-8}" \
  --record-fps "${FPS:-5}" \
  --csv-image-format wide \
  --no-video \
  --seed "${SEED:-10}" \
  --output-dir "$OUT"
echo "Wide CSV written to: $OUT/dataset.csv.gz"

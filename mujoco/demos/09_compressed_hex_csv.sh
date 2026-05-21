#!/usr/bin/env bash
# Dataset feature demo: compressed CSV with one compact 512-character hex field for each 16x16 uint8 grayscale image.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/09_compressed_hex_csv}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-12}" \
  --record-fps "${FPS:-10}" \
  --csv-image-format hex \
  --no-video \
  --seed "${SEED:-9}" \
  --output-dir "$OUT"
echo "Compressed CSV written to: $OUT/dataset.csv.gz"

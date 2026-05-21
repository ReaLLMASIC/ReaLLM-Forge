#!/usr/bin/env bash
# Throughput-oriented data demo: lower render resolution, lower sample FPS, no MP4, compressed CSV only.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/12_csv_only_fast_collection}"
run_collector \
  --gl "${GL:-egl}" \
  --num-episodes "${EPISODES:-16}" \
  --num-workers "${WORKERS:-4}" \
  --duration "${DURATION:-15}" \
  --width 160 \
  --height 120 \
  --record-fps 10 \
  --no-video \
  --csv-image-format hex \
  --seed "${SEED:-12}" \
  --output-dir "$OUT"
echo "Compressed CSV written to: $OUT/dataset.csv.gz"

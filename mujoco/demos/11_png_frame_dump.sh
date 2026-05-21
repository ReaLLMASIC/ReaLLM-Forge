#!/usr/bin/env bash
# Image dump demo: save occasional full-resolution RGB first-person PNG frames in addition to video/CSV.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/11_png_frame_dump}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-10}" \
  --record-fps "${FPS:-15}" \
  --save-frame-images-every "${SAVE_FRAME_IMAGES_EVERY:-10}" \
  --camera-height-above-top 0.06 \
  --camera-pitch-deg 20 \
  --seed "${SEED:-11}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"
echo "PNG frames are under: $OUT/frames/"

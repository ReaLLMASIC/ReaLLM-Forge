#!/usr/bin/env bash
# Video option demo: save clean first-person MP4 frames with no action/timestamp overlay.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/15_no_annotations_video}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-12}" \
  --no-annotate-video \
  --camera-height-above-top 0.05 \
  --camera-pitch-deg 15 \
  --seed "${SEED:-15}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

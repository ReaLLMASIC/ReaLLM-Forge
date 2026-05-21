#!/usr/bin/env bash
# Basic feature demo: random wandering, bump behavior, first-person MP4, compressed CSV, generated MJCF XML.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/01_basic_headless_video_csv}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-20}" \
  --seed "${SEED:-1}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

#!/usr/bin/env bash
# GPU/offscreen rendering demo for Linux GPU machines. Uses MuJoCo EGL headless rendering.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/04_gpu_egl_headless}"
run_collector \
  --gl "${GL:-egl}" \
  --duration "${DURATION:-30}" \
  --width "${WIDTH:-640}" \
  --height "${HEIGHT:-480}" \
  --record-fps "${FPS:-30}" \
  --seed "${SEED:-4}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

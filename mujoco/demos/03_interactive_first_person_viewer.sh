#!/usr/bin/env bash
# Viewer feature demo: interactive MuJoCo viewer locked to the Roomba first-person camera.
# On macOS, try: PYTHON=mjpython scripts/03_interactive_first_person_viewer.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/03_interactive_first_person_viewer}"
run_collector \
  --view \
  --gl "${GL:-glfw}" \
  --view-camera roomba_fp \
  --duration "${DURATION:-30}" \
  --camera-height-above-top "${CAMERA_HEIGHT:-0.05}" \
  --camera-pitch-deg "${CAMERA_PITCH:-15}" \
  --seed "${SEED:-3}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

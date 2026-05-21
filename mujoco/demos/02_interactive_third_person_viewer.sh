#!/usr/bin/env bash
# Viewer feature demo: interactive MuJoCo viewer using a third-person free camera.
# On macOS, try: PYTHON=mjpython scripts/02_interactive_third_person_viewer.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/02_interactive_third_person_viewer}"
run_collector \
  --view \
  --gl "${GL:-glfw}" \
  --view-camera free \
  --duration "${DURATION:-30}" \
  --seed "${SEED:-2}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

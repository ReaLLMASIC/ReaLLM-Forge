#!/usr/bin/env bash
# Geometry/config demo: larger room, taller walls, slightly larger/heavier robot, custom camera field of view.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/13_custom_room_robot_geometry}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-20}" \
  --room-size 6.0 \
  --wall-height 0.55 \
  --wall-thickness 0.08 \
  --robot-radius 0.19 \
  --robot-height 0.10 \
  --robot-mass 4.2 \
  --camera-fovy 105 \
  --camera-height-above-top 0.08 \
  --camera-pitch-deg 25 \
  --seed "${SEED:-13}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

#!/usr/bin/env bash
# Bump feature demo: small room + fixed start makes the robot hit a wall, back up, then turn roughly 180 degrees.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/07_bump_reverse_180}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-12}" \
  --fixed-start \
  --room-size 1.0 \
  --speed 0.40 \
  --reverse-speed 0.28 \
  --reverse-seconds 0.85 \
  --turn-interval-mean 999 \
  --turn-interval-std 0 \
  --bump-turn-std-deg 4 \
  --camera-height-above-top 0.05 \
  --camera-pitch-deg 12 \
  --seed "${SEED:-7}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

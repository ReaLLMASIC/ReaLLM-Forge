#!/usr/bin/env bash
# Motion-control demo: faster drive/turn speeds, stronger controller, and lower command noise.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/14_motion_control_hyperparams}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-20}" \
  --speed 0.55 \
  --reverse-speed 0.35 \
  --turn-speed 2.2 \
  --control-kv 180 \
  --max-force 100 \
  --max-torque 25 \
  --cmd-noise-v-std 0.005 \
  --cmd-noise-omega-std 0.010 \
  --seed "${SEED:-14}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

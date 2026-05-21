#!/usr/bin/env bash
# Behavior randomization demo: more frequent Gaussian-randomized turns with larger angle variance.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/06_gaussian_random_wander}"
run_collector \
  --gl "${GL:-auto}" \
  --duration "${DURATION:-25}" \
  --turn-interval-mean 1.2 \
  --turn-interval-std 0.45 \
  --wander-turn-mean-deg 55 \
  --wander-turn-std-deg 30 \
  --min-wander-turn-deg 10 \
  --cmd-noise-v-std 0.025 \
  --cmd-noise-omega-std 0.06 \
  --seed "${SEED:-6}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

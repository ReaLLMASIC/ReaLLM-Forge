#!/usr/bin/env bash
# First-person camera hyperparameter demo: run several camera heights and downward/upward pitch angles.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
BASE="${OUT:-$ROOT/runs/05_camera_height_pitch_sweep}"
GL_BACKEND="${GL:-auto}"
DUR="${DURATION:-8}"
run_collector --gl "$GL_BACKEND" --duration "$DUR" --seed 50 --camera-height-above-top 0.025 --camera-pitch-deg 0  --output-dir "$BASE/h025_pitch000"
run_collector --gl "$GL_BACKEND" --duration "$DUR" --seed 51 --camera-height-above-top 0.060 --camera-pitch-deg 20 --output-dir "$BASE/h060_pitch020"
run_collector --gl "$GL_BACKEND" --duration "$DUR" --seed 52 --camera-height-above-top 0.100 --camera-pitch-deg 35 --output-dir "$BASE/h100_pitch035"
run_collector --gl "$GL_BACKEND" --duration "$DUR" --seed 53 --camera-height-above-top 0.060 --camera-pitch-deg -10 --output-dir "$BASE/h060_pitch_minus010"
echo "Outputs written under: $BASE"

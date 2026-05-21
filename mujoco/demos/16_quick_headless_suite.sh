#!/usr/bin/env bash
# Runs a small non-viewer subset. Good for validating a machine without opening windows.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION="${DURATION:-5}" GL="${GL:-auto}" "$SCRIPT_DIR/01_basic_headless_video_csv.sh"
DURATION="${DURATION:-5}" GL="${GL:-auto}" "$SCRIPT_DIR/05_camera_height_pitch_sweep.sh"
DURATION="${DURATION:-5}" GL="${GL:-auto}" "$SCRIPT_DIR/07_bump_reverse_180.sh"
DURATION="${DURATION:-5}" GL="${GL:-auto}" "$SCRIPT_DIR/09_compressed_hex_csv.sh"
DURATION="${DURATION:-5}" GL="${GL:-auto}" "$SCRIPT_DIR/10_wide_csv_pixels.sh"

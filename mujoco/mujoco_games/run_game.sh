#!/usr/bin/env bash
# Runs the snake-inspired pink target detection and dataset logging routine.

OUT="${OUT:-runs/nanogpt_snake_seek}"

# ----------------- Settable Target Bounds -----------------
CEILING_HEIGHT="${CEILING_HEIGHT:-3.048}"   # 10 Feet Default
CAMERA_HEIGHT="${CAMERA_HEIGHT:-0.07}"      # Set to 0.07m above robot top
CAMERA_PITCH="${CAMERA_PITCH:-18}"          # Downward angle to spot target spheres close by
DURATION="${DURATION:-45}"                  # Run duration per episode

# --- High-Resolution Variables ---
VIDEO_WIDTH="1280"
VIDEO_HEIGHT="720"
# ----------------------------------------------------------

echo "Initializing Target Seeking Collection Loop..."
echo " - Resolution: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}"
echo " - Camera height relative to deck: $CAMERA_HEIGHT meters"
echo " - Enclosure Ceiling height:       $CEILING_HEIGHT meters"

python3 roomba_mujoco_collect.py \
  --output-dir "$OUT" \
  --duration "$DURATION" \
  --ceiling-height "$CEILING_HEIGHT" \
  --camera-height-above-top "$CAMERA_HEIGHT" \
  --camera-pitch-deg "$CAMERA_PITCH" \
  --width "$VIDEO_WIDTH" \
  --height "$VIDEO_HEIGHT" \
  --num-episodes 1 \
  --record-fps 30 \
  --nanogpt-csv \
  --view-camera roomba_fp

echo "Dataset outputs generated successfully at: $OUT"

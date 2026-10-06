#!/usr/bin/env bash
# Usage: bash split_wav.sh INPUT_WAV [OUTPUT_PREFIX]
# FFmpeg segments at packet boundaries, so segment counts/durations are approximate.
set -euo pipefail
INPUT_FILE="${1:?Usage: bash split_wav.sh INPUT_WAV [OUTPUT_PREFIX]}"
PREFIX="${2:-${INPUT_FILE%.*}}"
TOTAL_DURATION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT_FILE")"
SEGMENT_TIME="$(python3 - "$TOTAL_DURATION" <<'PY'
import math,sys
seconds=float(sys.argv[1])
if not math.isfinite(seconds) or seconds <= 0:
    raise SystemExit('Input duration must be positive and finite')
print(format(seconds/10, '.12g'))
PY
)"
ffmpeg -nostdin -v warning -n -i "$INPUT_FILE" -f segment -segment_time "$SEGMENT_TIME" \
  -c copy "${PREFIX}_part_%02d.wav"
echo "Completed segments with prefix: $PREFIX"

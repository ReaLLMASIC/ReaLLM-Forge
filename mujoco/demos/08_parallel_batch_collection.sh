#!/usr/bin/env bash
# Parallel collection demo: multiple independent episodes, multiprocessing, videos only for selected episodes.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
OUT="${OUT:-$ROOT/runs/08_parallel_batch_collection}"
run_collector \
  --gl "${GL:-egl}" \
  --num-episodes "${EPISODES:-8}" \
  --num-workers "${WORKERS:-4}" \
  --duration "${DURATION:-20}" \
  --video-every "${VIDEO_EVERY:-4}" \
  --record-fps "${FPS:-15}" \
  --seed "${SEED:-8}" \
  --output-dir "$OUT"
echo "Outputs written to: $OUT"

#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_single_gpu_process

BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-5}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/${HARDWARE_PROFILE}_benchmark_$(run_stamp)_$$}"

exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode benchmark \
  --hardware-profile "$HARDWARE_PROFILE" \
  --benchmark-warmup "$BENCHMARK_WARMUP" \
  --benchmark-steps "$BENCHMARK_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

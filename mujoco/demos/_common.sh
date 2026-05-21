#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
mkdir -p "$ROOT/runs"
cd "$ROOT"
run_collector() {
  "$PYTHON" "$ROOT/roomba_mujoco_collect.py" "$@"
}

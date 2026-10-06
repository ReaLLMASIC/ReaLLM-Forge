#!/usr/bin/env bash
# Usage: bash data/mel_mc_int/run.sh INPUT_AUDIO [OUTPUT_ROOT]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/pipeline.py" single "$@"

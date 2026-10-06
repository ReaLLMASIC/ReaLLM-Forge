#!/usr/bin/env bash
# Usage: bash data/mel_mc_int/demo_infer.sh OUT_DIR INPUT_AUDIO CUTOFF_SECONDS [FRAMES] [--manifest PATH]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/pipeline.py" infer "$@"

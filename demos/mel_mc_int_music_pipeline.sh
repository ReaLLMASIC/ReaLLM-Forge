#!/usr/bin/env bash
# Encode a recording-level train/validation split, train, sample, and reconstruct.
# Usage: bash demos/mel_mc_int_music_pipeline.sh MUSIC_DIR [PROMPT_AUDIO] [CUTOFF_SECONDS]
# Settings and defaults: data/mel_mc_int/README.md
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$REPO_ROOT/data/mel_mc_int/pipeline.py" folder "$@"

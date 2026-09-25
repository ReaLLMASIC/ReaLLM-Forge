#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG="${1:-configs/h100_pilot.json}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python cpt.py preflight --config "$CONFIG"
python cpt.py run --config "$CONFIG"

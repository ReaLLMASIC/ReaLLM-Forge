#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG="${1:-configs/h100_math_pilot.json}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c 'import lm_eval; print("Evaluation harness is installed")'
python cpt.py preflight --config "$CONFIG"
python task_sft.py run --config "$CONFIG"
python benchmarks.py suite --config "$CONFIG"

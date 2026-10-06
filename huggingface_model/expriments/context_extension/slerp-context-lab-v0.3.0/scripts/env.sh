#!/usr/bin/env bash
# Source this file; all active files stay beside the extracted package.
SLERP_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SLERP_WORK_ROOT="$SLERP_PACKAGE_ROOT"
export HF_HOME="$SLERP_WORK_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="$SLERP_PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SLERP_PYTHON="${SLERP_PYTHON:-$SLERP_WORK_ROOT/.venv/bin/python}"
export SLERP_PYTHON


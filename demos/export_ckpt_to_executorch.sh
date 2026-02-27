#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "" ]]; then
  echo "Usage: $0 <ckpt-path> [pte-path]"
  exit 1
fi

CKPT_PATH=$1
PTE_PATH=${2:-}

if [[ -n "$PTE_PATH" ]]; then
  python -m model_exports.executorch.export_checkpoint --ckpt "$CKPT_PATH" --pte-path "$PTE_PATH" "${@:3}"
else
  python -m model_exports.executorch.export_checkpoint --ckpt "$CKPT_PATH" "${@:2}"
fi

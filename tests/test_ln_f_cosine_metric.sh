#!/bin/bash
# test_ln_f_cosine_metric.sh
# Runs a minimal training loop and verifies ln_f cosine metric is produced.

set -euo pipefail

# Run from repo root
pushd "$(dirname "$0")/.." >/dev/null

LOGFILE=$(mktemp)

python train.py \
    --device cpu \
    --compile False \
    --max_iters 1 \
    --eval_iters 1 \
    --eval_interval 1 \
    --block_size 8 \
    --batch_size 4 \
    --n_layer 1 \
    --n_head 1 \
    --n_embd 32 \
    >"$LOGFILE" 2>&1

grep -E "LnFcos:" "$LOGFILE"
grep -E "LnFcos95:" "$LOGFILE"

popd >/dev/null

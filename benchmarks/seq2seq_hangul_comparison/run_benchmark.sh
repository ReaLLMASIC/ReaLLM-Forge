#!/usr/bin/env bash
set -e

# Benchmark runner for Three-Hot vs Hangul Factorizer on Seq2Seq English-to-Korean translation
VENV_PYTHON="/usr/local/google/home/kahye/ReaLLM-Forge/venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

echo "================================================================="
echo "Seq2Seq Benchmark: Three-Hot Tokenizer vs. Hangul Factorizer"
echo "Repository Root: ${REPO_ROOT}"
echo "Using Python: ${VENV_PYTHON}"
echo "================================================================="

# Mode 1: Fast Pilot Test (verifies pipeline end-to-end on 2,000 samples, 2 epochs)
if [ "$1" == "--pilot" ]; then
    echo "Running Pilot Test (2 epochs, 2k samples)..."
    ${VENV_PYTHON} ${SCRIPT_DIR}/run_benchmark.py \
        --data_dir "data/korean_seq2seq_bench" \
        --output_dir "out_seq2seq_bench_pilot" \
        --epochs 2 \
        --batch_size 32 \
        --max_train_samples 2000 \
        --max_val_samples 200 \
        --max_test_samples 200
    exit 0
fi

# Mode 2: Full Benchmark Run
EPOCHS=${1:-15}
BATCH_SIZE=${2:-32}

echo "Running Full Benchmark (${EPOCHS} epochs, batch size ${BATCH_SIZE})..."
${VENV_PYTHON} ${SCRIPT_DIR}/run_benchmark.py \
    --data_dir "data/korean_seq2seq_bench" \
    --output_dir "out_seq2seq_bench" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr 1e-4

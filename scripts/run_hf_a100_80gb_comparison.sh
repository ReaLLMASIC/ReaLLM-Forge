#!/usr/bin/env bash
set -euo pipefail

# Matched ~1.31B-token runs for one NVIDIA A100 80GB:
# 10,000 * 32 sequences * 1,024 tokens * 4 accumulation steps.
# Run from any directory; outputs are written below the repository by default.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python scripts/compare_hf_relu2max.py \
  --dataset roneneldan/TinyStories \
  --tokenizer gpt2 \
  --train-samples 0 \
  --eval-samples 10000 \
  --pack-text \
  --block-size 1024 \
  --hidden-size 768 \
  --intermediate-size 3072 \
  --layers 12 \
  --heads 12 \
  --kv-heads 12 \
  --batch-size 32 \
  --gradient-accumulation-steps 4 \
  --max-steps 10000 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --warmup-steps 500 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --save-total-limit 2 \
  --bf16 \
  --tf32 \
  --dataloader-workers 8 \
  --relu2max-accelerator auto \
  --lm-eval-batch-size auto \
  --lm-eval-tasks lambada_openai hellaswag piqa winogrande arc_easy \
  --output-dir runs/a100-80gb-relu2max-vs-softmax \
  "$@"

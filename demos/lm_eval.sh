#!/bin/bash
# demos/lm_eval.sh

python3 -m pip install lm-eval

# Train a model with benchmarking evaluation when the validation loss improves
## max tokens of 500 when benchmarking
## applies the arc_easy and hellaswag tasks
python3 train.py \
    --out_dir out_lm_eval \
    --dataset wikitext103 \
    --lm_eval_tasks arc_easy,hellaswag \
    --max_benchmark_tokens 500 \
    --no-benchmark_each_eval

# Apply the same benchmarks to the model after training
python3 sample.py \
    --out_dir out_lm_eval \
    --lm_eval_tasks arc_easy,hellaswag \
    --batch_size 1
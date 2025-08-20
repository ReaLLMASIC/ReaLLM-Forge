# `benchmarks/`

This directory contains scripts for running benchmarks and evaluating model performance.

## Key Files and Their Purpose

| File/Folder | Description |
|---|---|
| `__init__.py` | Initializes the benchmarks module. |
| `bench.py` | A script for running benchmarks. |
| `bleu.py` | A script for calculating the BLEU score, a metric for evaluating the quality of machine-translated text. |
| `dataset_metrics.py` | A script for calculating various metrics on a dataset, such as perplexity and accuracy. |
| `gpt_lm_eval_wrapper.py` | A wrapper for using the `lm-evaluation-harness` with the model. This allows for standardized benchmarking on a wide range of tasks. |
| `softmax_sweep.py` | A script for sweeping through different softmax functions and evaluating their performance. |
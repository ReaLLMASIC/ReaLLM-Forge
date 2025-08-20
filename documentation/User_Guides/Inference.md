# 💡 Generating Samples (Inference)

This guide explains how to use the `sample.py` script to generate text from a trained model, run benchmarks, and use advanced visualization features.

## Basic Inference

To generate text from a trained model, you need to specify the output directory of your training run.

```bash
python sample.py --out_dir=out/<your_run_name>
```

### Key Inference Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--out_dir` | str | 'out' | The directory containing the `ckpt.pt` file from your training run. |
| `--start` | str | "\n" | The initial prompt for the model. You can provide a string directly or specify a file with `FILE:prompt.txt`. |
| `--num_samples` | int | 3 | The number of independent samples to generate. |
| `--max_new_tokens` | int | 500 | The maximum number of tokens to generate for each sample. |
| `--temperature` | float | 0.8 | Controls the randomness of the output. Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.7) make it more deterministic. |
| `--top_k` | int | 200 | Restricts the sampling to the `k` most likely next tokens. |
| `--seed` | int | 1337 | The random seed for reproducibility. |

## Interactive Mode

For a more interactive experience, you can use the `--interactive` flag. This will allow you to have a conversation with the model.

```bash
python sample.py --out_dir=out/<your_run_name> --interactive
```

## Visualization and Analysis

The `sample.py` script includes several features for visualizing and analyzing the model's output.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--colorize_output` | bool | False | Color-codes the generated tokens based on the model's confidence. |
| `--colorize_mode` | str | 'minmax' | The mode for colorizing the output. Options are 'minmax', 'softmax', 'softmax_top_k', 'rank', 'dot_product', 'all'. |
| `--show_heatmaps` | bool | False | Generates a heatmap for each generated token, showing the probability distribution over the top-k vocabulary tokens. |
| `--chart_type` | str | 'heatmap' | The type of chart to display for `--show_heatmaps`. Options are 'heatmap' and 'barchart'. |
| `--last_k_tokens` | int | 10 | The number of recent tokens to display in the heatmap's context label. |

## Benchmarking

The `sample.py` script is also integrated with the `lm-evaluation-harness` for standardized benchmarking.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lm_eval_tasks` | str | None | A comma-separated list of tasks to evaluate the model on (e.g., `hellaswag,arc_easy`). |
| `--lm_eval_results_output` | str | None | The file to save the lm-eval results to. |
| `--batch_size` | int | 1 | The batch size to use for evaluation. |

```bash
python sample.py --out_dir=out/<your_run_name> --lm_eval_tasks="hellaswag,arc_easy"
```

This will run the specified benchmarks and save the results to a JSON file in the output directory.

```
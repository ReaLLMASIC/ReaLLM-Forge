# 3.3 In-Training Analysis & Sampling

This section provides a comprehensive guide to all arguments related to the analysis and sampling features that can be used during the training process. This is a powerful tool for qualitatively assessing the model's progress during training.

## In-Training Sampling

These arguments control the generation of samples from the model at evaluation time.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--max_sample_tokens` | int | None | If specified, the maximum number of tokens to sample and print after each validation loss. |
| `--sample_each_eval` | bool | False | If `True`, a sample will be generated at every evaluation interval, regardless of whether the validation loss has improved. This is useful for observing how the model's generations change over time, even if it is overfitting. |
| `--sample_start_tokens` | str | '\n' | The starting tokens for the sample generation. |
| `--sample_only` | bool | False | If `True`, the script will only run the sampling process and then exit. This is useful for quickly generating samples from a trained model without having to run the full training script. |
| `--dataset_benchmarks` | bool | False | If `True`, the script will run dataset benchmark metrics on a random slice of the dataset after each validation. |
| `--sample_metrics` | bool | False | If `True`, the script will display sample metrics like spelling correctness during sampling. |

## In-Training Visualization

These arguments control the visualization of the model's output during in-training sampling.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--colorize_output` | bool | True | Whether to colorize the tokens in the generated sample based on their predicted probabilities. This can provide a quick and intuitive way to assess the model's confidence. |
| `--colorize_mode` | str | 'minmax' | The mode for colorizing the output. |
| `--show_heatmaps` | bool | False | Whether to show heatmaps (or bar charts) of the top-k token probabilities. This is a powerful tool for understanding the model's decision-making process at each step of the generation. |
| `--chart_type` | str | 'heatmap' | The type of chart to display if `--show_heatmaps` is set. |
| `--last_k_tokens` | int | 10 | The number of recent tokens to display in the heatmap's context label. |
| `--sample_file` | str | "sample.txt" | The output file for the inference samples. |
| `--token_boundary` | str | None | An optional separator string to place between emitted tokens. |
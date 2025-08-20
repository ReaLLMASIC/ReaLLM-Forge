# 3.2 Metric Logging

This section provides a comprehensive guide to all arguments that control which metrics are logged during training.

## Core Metric Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_interval` | int | 10 | The number of iterations between logging training metrics to the console. |

## Advanced Metric Logging

These arguments allow you to log more detailed metrics about the training process.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_btc_train` | bool | False | Whether to log better-than-chance training metrics. This is a measure of how much better the model is doing than a random baseline. |
| `--log_btc_per_param` | bool | False | Whether to log better-than-chance-per-parameter metrics. This is a measure of the model's efficiency. |
| `--log_grad_norm` | bool | False | Whether to log the gradient norm. This can be useful for debugging training instability. |
| `--log_grad_std` | bool | False | Whether to log the gradient standard deviation. This can also be useful for debugging training instability. |
| `--log_all_metrics` | bool | False | A shorthand to enable logging of all metrics. |

## Model Statistics

These arguments control the generation of a detailed table of statistics for the model's weights and activations.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--compute_model_stats` | bool | False | Whether to compute and print the model statistics table during evaluation. This can be slow, so it is disabled by default. |
| `--model_stats_device` | str | 'gpu' | The device to use for aggregating the statistics. 'gpu' is faster, but 'cpu' can be used if you are running out of GPU memory. |
| `--print_model_stats_table` | str | None | If specified, the model statistics table will be saved to this CSV file. |
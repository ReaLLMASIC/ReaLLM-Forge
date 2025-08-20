# 📝 Logging

This document explains how to configure and use the logging features in the ReaLLM-Forge repository.

## TensorBoard Logging

TensorBoard is the primary logging backend for visualizing training metrics. It is enabled by default.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--tensorboard_log` | bool | True | Whether to use TensorBoard for logging. |
| `--tensorboard_log_dir` | str | 'logs' | The directory to save TensorBoard logs. |
| `--tensorboard_run_name` | str | None | The name of the TensorBoard run. If not specified, a timestamp is used. |
| `--tensorboard_graph` | bool | True | Whether to log the model graph to TensorBoard. |

## CSV Logging

You can also log training metrics to a CSV file. This is useful for programmatic analysis of experiment results.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv_log` | bool | True | Whether to log metrics to a CSV file. |
| `--csv_dir` | str | 'csv_logs' | The directory to save CSV logs. |
| `--csv_name` | str | 'output' | The basename for the output CSV file. |

## Metric Logging Toggles

These arguments allow you to control which metrics are logged.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log_btc_train` | bool | False | Whether to log better-than-chance training metrics. |
| `--log_btc_per_param` | bool | False | Whether to log better-than-chance-per-parameter metrics. |
| `--log_grad_norm` | bool | False | Whether to log the gradient norm. |
| `--log_grad_std` | bool | False | Whether to log the gradient standard deviation. |
| `--log_all_metrics` | bool | False | A shorthand to enable logging of all metrics. |

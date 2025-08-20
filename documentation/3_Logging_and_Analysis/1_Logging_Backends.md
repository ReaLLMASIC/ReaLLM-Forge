# 3.1 Logging Backends

This section provides a comprehensive guide to all arguments related to the configuration of the logging backends.

## TensorBoard

TensorBoard is the primary logging backend for visualizing training metrics. It is enabled by default.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--tensorboard_log` | bool | True | A global switch to enable or disable TensorBoard logging. |
| `--tensorboard_log_dir` | str | 'logs' | The directory to save TensorBoard logs. |
| `--tensorboard_run_name` | str | None | The name of the TensorBoard run. If not specified, a timestamp is used. This is useful for organizing your experiments. |
| `--tensorboard_graph` | bool | True | Whether to log the model graph to TensorBoard. This can be useful for visualizing the model architecture, but can also be slow and consume a lot of disk space. |

## CSV

You can also log training metrics to a CSV file. This is useful for programmatic analysis of experiment results.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--csv_log` | bool | True | A global switch to enable or disable CSV logging. |
| `--csv_dir` | str | 'csv_logs' | The directory to save CSV logs. |
| `--csv_name` | str | 'output' | The basename for the output CSV file. |
| `--csv_ckpt_dir` | str | '' | A subdirectory within `csv_dir` to save the CSV file to. This is useful for organizing your CSV logs by checkpoint. |

## Weights & Biases (W&B)

Weights & Biases is a third-party logging service that provides a more powerful and interactive way to visualize and analyze your experiments.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wandb_log` | bool | False | A global switch to enable or disable W&B logging. |
| `--wandb_project` | str | 'out-test' | The W&B project to log to. |
| `--wandb_run_name` | str | 'logs-test' | The name of the W&B run. |
# 🔬 Analysis

This document explains how to use the analysis tools in the ReaLLM-Forge repository.

## Model Statistics Table

The `train.py` script can generate a detailed table of statistics for the model's weights and activations. This is a powerful tool for debugging and analyzing model behavior.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--compute_model_stats` | bool | False | Whether to compute and print the model statistics table during evaluation. |
| `--model_stats_device` | str | 'gpu' | The device to use for aggregating the statistics. Options are 'cpu' and 'gpu'. |
| `--print_model_stats_table` | str | None | If specified, the model statistics table will be saved to this CSV file. |

To view the generated CSV file, you can use the `view_model_stats.py` script.

```bash
python view_model_stats.py <path_to_csv_file>
```

## Checkpoint Inspector

The `analysis/checkpoint_analysis/inspect_ckpts.py` script can be used to inspect the checkpoints in an output directory and find the best validation loss.

```bash
python analysis/checkpoint_analysis/inspect_ckpts.py --directory <your_out_dir>
```

This script will print a table of all the checkpoints in the directory, sorted by validation loss. This is useful for monitoring the progress of a training run and for identifying the best model to use for inference.

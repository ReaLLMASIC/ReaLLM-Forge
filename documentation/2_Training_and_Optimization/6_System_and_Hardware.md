# 2.6 System & Hardware

This section provides a comprehensive guide to all arguments related to the system-level and hardware configuration of the training run.

## Core System Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--device` | str | 'cuda' | The device to use for training. Options are 'cuda', 'cpu', 'cuda:0', etc. |
| `--dtype` | str | "float16" | The data type to use for training. "bfloat16" is recommended for modern GPUs, as it provides a good balance between precision and performance. "float16" can be faster, but is more prone to numerical instability. "float32" is the most stable, but also the slowest. |
| `--compile` | bool | False | If `True`, compiles the model with `torch.compile` for a significant speed-up. This is highly recommended for modern GPUs. |

## Distributed Data Parallel (DDP)

These arguments are used for distributed training across multiple GPUs.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--backend` | str | 'nccl' | The backend to use for distributed training. 'nccl' is the recommended backend for NVIDIA GPUs. |
# `analysis/`

This directory contains scripts for analyzing model checkpoints, activations, and other aspects of the training process.

## Key Files and Their Purpose

| File/Folder | Description |
|---|---|
| `activation_analysis/` | Contains scripts for analyzing the activations of the model. This can be useful for understanding how the model is representing information and for debugging training issues. |
| `checkpoint_analysis/inspect_ckpts.py` | A script for inspecting the checkpoints in an output directory and finding the best validation loss. This is the primary tool for monitoring the progress of a training run. |
| `compression_algorithms/` | Contains scripts for exploring compression algorithms. |
| `distillation/` | Contains scripts for model distillation. |
| `tokenizer_analysis/` | Contains scripts for analyzing tokenizers. |
| `vector_distribution/` | Contains scripts for analyzing the distribution of vectors, such as embeddings or activations. |
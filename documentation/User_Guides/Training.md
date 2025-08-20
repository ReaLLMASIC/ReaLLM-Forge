# 🏋️ Training Models

This guide provides a comprehensive overview of how to use the `train.py` script to train models in the ReaLLM-Forge repository.

## Basic Training

The most straightforward way to train a model is to specify a dataset and an output directory. The script will use the default hyperparameters defined in `train_args.py`.

```bash
python train.py --dataset=<your_dataset> --out_dir=out/<your_run_name>
```

### Key Training Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | 'shakespeare_char' | The name of the dataset directory in `data/`. |
| `--out_dir` | str | 'out' | The directory to save checkpoints and logs. |
| `--batch_size` | int | 64 | The number of sequences to process in parallel. |
| `--max_iters` | int | 3500 | The total number of training iterations to run. |
| `--learning_rate` | float | 1e-3 | The maximum learning rate. |
| `--eval_interval` | int | 250 | The number of iterations between evaluations on the validation set. |
| `--eval_iters` | int | 200 | The number of iterations to run for each evaluation. |
| `--log_interval` | int | 10 | The number of iterations between logging training metrics. |
| `--seed` | int | 1337 | The random seed for reproducibility. |

## Resuming Training

You can resume a previous training run from a checkpoint. This will restore the model, optimizer state, and training iteration number.

```bash
python train.py --init_from=resume --out_dir=out/<your_run_name>
```

## Fine-Tuning from a Pretrained Model

You can also initialize training from a pretrained GPT-2 model. This is useful for fine-tuning on a new dataset.

```bash
python train.py --init_from=gpt2 --dataset=<your_dataset> --out_dir=out/<your_finetune_run>
```

You can specify different GPT-2 model sizes with the `--gpt2_type` argument (e.g., `gpt2-medium`, `gpt2-large`, `gpt2-xl`).

## Multi-Dataset Training

The repository supports training on multiple datasets simultaneously. There are two modes for this:

-   **`multidataset`:** The script samples from a list of datasets, one at a time. You can control the sampling probabilities with `--dataset_sampling_probs`.
    ```bash
    python train.py --training_mode=multidataset --dataset_list <dataset1> <dataset2> --dataset_sampling_probs 0.5 0.5
    ```
-   **`multicontext`:** The script processes batches from multiple datasets in parallel. This is a more advanced feature.
    ```bash
    python train.py --training_mode=multicontext --multicontext_datasets <dataset1> <dataset2>
    ```

## Optimizer and Learning Rate Scheduler

You can select different optimizers and learning rate schedulers to customize the training process.

### Optimizer Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--optimizer` | str | "adamw" | The optimizer to use (e.g., `adamw`, `sgd`). |
| `--weight_decay` | float | 1e-1 | The weight decay for the optimizer. |
| `--beta1` | float | 0.9 | The beta1 parameter for Adam-style optimizers. |
| `--beta2` | float | 0.99 | The beta2 parameter for Adam-style optimizers. |
| `--grad_clip` | float | 1.0 | The value for gradient clipping. |

### Learning Rate Scheduler Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lr_scheduler` | str | "none" | The learning rate scheduler (e.g., `cosine`, `exponential`). |
| `--decay_lr` | bool | False | Whether to decay the learning rate. |
| `--warmup_iters` | int | 100 | The number of warmup iterations for the learning rate. |
| `--lr_decay_iters` | int | 3500 | The number of iterations over which to decay the learning rate. |
| `--min_lr` | float | 1e-4 | The minimum learning rate. |

For a full list of available optimizers, schedulers, and their arguments, please refer to `train_args.py`.
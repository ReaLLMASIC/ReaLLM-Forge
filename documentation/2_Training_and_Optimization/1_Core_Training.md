# 2.1 Core Training

These arguments control the fundamental parameters of the training run.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | 'shakespeare_char' | The name of the dataset directory in `data/`. This is a required argument. |
| `--out_dir` | str | 'out' | The directory to save checkpoints and logs. |
| `--batch_size` | int | 64 | The number of sequences to process in parallel in a single forward/backward pass. A larger batch size can lead to more stable gradients, but also requires more memory. |
| `--max_iters` | int | 3500 | The total number of training iterations to run. |
| `--eval_interval` | int | 250 | The number of iterations between evaluations on the validation set. |
| `--eval_iters` | int | 200 | The number of iterations to run for each evaluation to get a stable loss estimate. |
| `--log_interval` | int | 10 | The number of iterations between logging training metrics to the console. |
| `--seed` | int | 1337 | The random seed for reproducibility. |
| `--gradient_accumulation_steps` | int | 1 | The number of forward/backward passes to accumulate gradients over before performing an optimizer step. The effective batch size is `batch_size * gradient_accumulation_steps`. This is a useful technique for increasing the batch size when you are limited by GPU memory. |
| `--patience` | int | None | If set, training will stop if the validation loss has not improved for this many evaluation intervals. This is a form of early stopping. |
| `--focus_on_top1_loss` | bool | False | If `True`, uses a custom loss function that puts more emphasis on the top-1 prediction. This can be useful for tasks where the top-1 accuracy is the most important metric. |
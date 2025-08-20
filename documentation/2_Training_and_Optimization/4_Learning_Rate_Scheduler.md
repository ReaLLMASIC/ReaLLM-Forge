# 2.4 Learning Rate Scheduler

This section provides a comprehensive guide to all arguments related to the learning rate scheduler. The learning rate scheduler is responsible for adjusting the learning rate during training, which can have a significant impact on the model's performance.

## Core Scheduler Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lr_scheduler` | str | "none" | The learning rate scheduler to use. "none" means a constant learning rate. "cosine" uses a cosine annealing schedule. "exponential" uses an exponential decay schedule. "step" uses a step decay schedule. "plateau" reduces the learning rate when a metric has stopped improving. |
| `--learning_rate` | float | 1e-3 | The maximum learning rate. |
| `--min_lr` | float | 1e-4 | The minimum learning rate. |
| `--warmup_iters` | int | 100 | The number of warmup iterations, during which the learning rate increases linearly from 0 to `learning_rate`. This can help stabilize training in the early stages. |
| `--lr_decay_iters` | int | 3500 | The number of iterations over which to decay the learning rate. |
| `--lr_decay_match_max_iters` | bool | True | If `True`, `lr_decay_iters` will be set to `max_iters`. |

<br>

---

## Scheduler Specific Arguments

<details>
<summary><b>Cosine Annealing</b></summary>

This scheduler gradually decreases the learning rate following a cosine curve.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cosine_t_max` | int | 1000 | The maximum number of iterations for the cosine annealing schedule. |
| `--cosine_eta_min` | float | 0 | The minimum learning rate for the cosine annealing schedule. |
</details>

<details>
<summary><b>Exponential Decay</b></summary>

This scheduler decreases the learning rate by a multiplicative factor at each step.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--exponential_gamma` | float | 0.9 | The multiplicative factor for the exponential decay. |
</details>

<details>
<summary><b>Step Decay</b></summary>

This scheduler decreases the learning rate by a multiplicative factor at fixed intervals.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--step_lr_size` | int | 1000 | The number of iterations before decaying the learning rate. |
| `--step_lr_gamma` | float | 0.1 | The multiplicative factor for the step decay. |
</details>

<details>
<summary><b>Reduce on Plateau</b></summary>

This scheduler reduces the learning rate when a metric (typically the validation loss) has stopped improving.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--plateau_mode` | str | "min" | The mode for the plateau scheduler. "min" means the learning rate will be reduced when the metric stops decreasing. "max" means the learning rate will be reduced when the metric stops increasing. |
| `--plateau_factor` | float | 0.1 | The factor by which the learning rate is reduced. |
| `--plateau_patience` | int | 10 | The number of epochs with no improvement before reducing the learning rate. |
</details>
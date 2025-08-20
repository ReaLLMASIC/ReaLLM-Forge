# 2.3 Optimizer

This section provides a comprehensive guide to all arguments related to the optimizer.

## Core Optimizer Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--optimizer` | str | "adamw" | The optimizer to use for training. A wide variety of optimizers are available, including standard ones like `adamw` and `sgd`, as well as more experimental ones. |
| `--weight_decay` | float | 1e-1 | The weight decay for the optimizer. This is a form of regularization that penalizes large weights. |
| `--grad_clip` | float | 1.0 | The value for gradient clipping. Gradient clipping is a technique to prevent exploding gradients by capping the maximum value of the gradients. A value of 0.0 means no gradient clipping. |

<br>

---

## Optimizer Specific Arguments

<details>
<summary><b>AdamW</b></summary>

AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient update. It is the default optimizer in this repository.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--adamw_betas` | list | [0.9, 0.999] | The beta parameters for the AdamW optimizer, which control the exponential decay rates for the first and second moment estimates. |
| `--adamw_eps` | float | 1e-8 | The epsilon parameter for the AdamW optimizer, which is a small constant to prevent division by zero. |
</details>

<details>
<summary><b>AdamW with Activation Regularization</b></summary>

This is an experimental variant of AdamW that adds a regularization term for the activations.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--activation_decay` | float | 0.0 | The L2 regularization coefficient for activations. |
| `--activation_stat` | str | "stdev" | The statistic to use for modulating the activation regularization. Options are "stdev", "kurtosis", "max", "min", "abs_max". |
</details>

<details>
<summary><b>SGD</b></summary>

Stochastic Gradient Descent (SGD) is a classic optimization algorithm.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--sgd_momentum` | float | 0.9 | The momentum for the SGD optimizer. |
| `--sgd_nesterov` | bool | False | Whether to use Nesterov momentum. |
</details>

<details>
<summary><b>Adagrad</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--adagrad_lr_decay` | float | 0 | The learning rate decay for the Adagrad optimizer. |
</details>

<details>
<summary><b>RMSprop</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--rmsprop_alpha` | float | 0.99 | The smoothing constant for the RMSprop optimizer. |
</details>

... and many more. For a complete list of all available optimizers and their arguments, please refer to `train_args.py`.
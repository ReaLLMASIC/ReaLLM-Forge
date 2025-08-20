# 1.6 Softmax Functions

This section provides a comprehensive guide to all arguments that control the softmax function, which is used in both the attention mechanism and the final output layer.

## Core Softmax Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--softmax_variant_attn` | str | "softmax" | The softmax variant to use for the attention layers. |
| `--softmax_variant_output` | str | "softmax" | The softmax variant to use for the final output layer. |
| `--final_logit_softcapping` | bool | None | If specified, applies a softcapping function (like `tanh`) to the final logits before the softmax. This can prevent extreme values and stabilize training. |

<br>

---

## Softmax Variant Specific Arguments

<details>
<summary><b>ConSmax & SaturatingConSmax</b></summary>

These arguments control the ConSmax and SaturatingConSmax variants, which are designed to be more robust to outliers than the standard softmax.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--consmax_initial_beta` | float | 2.5 | The initial value for the beta parameter. |
| `--consmax_initial_gamma` | float | 100.0 | The initial value for the gamma parameter. |
| `--consmax_use_euler_base` | bool | True | Whether to use Euler's number as the base for the exponentiation. |
| `--consmax_base` | float | 2.0 | The base to use if `--consmax_use_euler_base` is `False`. |
| `--consmax_saturation` | float | 11.0 | The saturation point for SaturatingConSmax. |
| `--consmax_learnable_beta` | bool | True | Whether the beta parameter is learnable. |
| `--consmax_learnable_gamma` | bool | True | Whether the gamma parameter is learnable. |
</details>

<details>
<summary><b>ConSmaxV2</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--consmax_per_head` | bool | True | Whether to use different beta and gamma parameters for each attention head. |
| `--consmax_v2_clamping` | bool | False | Whether to clamp the input values. |
| `--consmax_v2_clamp_value` | float | 80.0 | The maximum value to clamp the inputs to. |
</details>

<details>
<summary><b>Polymax</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--polymax_x_intercept` | float | -100.0 | The x-intercept for the polynomial function. |
| `--polymax_y_intercept` | float | 1.0 | The y-intercept for the polynomial function. |
| `--polymax_power` | float | 2.0 | The power for the polynomial function. |
| `--polymax_divisor` | float | 1000.0 | The divisor for the polynomial function. |
</details>

<details>
<summary><b>Strongermax</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--strongermax_strength` | float | `math.e` | The strength parameter for the Strongermax function. |
| `--strongermax_div_by_sum_of_terms` | bool | True | Whether to divide by the sum of the terms. |
| `--strongermax_divisor` | float | 1.0 | The divisor for the Strongermax function. |
| `--strongermax_use_xmax` | bool | True | Whether to use the `x - x_max` normalization trick. |
</details>

<details>
<summary><b>PFLA-Softmax</b></summary>

These arguments control the Piecewise Fully Learnable Activation (PFLA) softmax variant.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pfla_softmax_num_points` | int | 30 | The number of inner control points. |
| `--pfla_softmax_left_bound` | float | -10.0 | The start of the x-range. |
| `--pfla_softmax_right_bound` | float | 10.0 | The end of the x-range. |
| `--pfla_softmax_learn_x` | bool | False | Whether to learn the x-positions of the knots. |
| `--pfla_softmax_learn_y` | bool | True | Whether to learn the y-values of the knots. |
| `--pfla_softmax_init_activation` | str | "gelu" | The reference activation function for initializing the knots. |
| `--pfla_softmax_density` | str | "linear" | The distribution of the x-knots. Options are "linear", "quad", "exp". |
| `--pfla_softmax_mode` | str | "linear" | The interpolation scheme. Options are "linear", "quadratic". |
</details>
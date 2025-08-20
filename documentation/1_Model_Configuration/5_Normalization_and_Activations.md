# 1.5 Normalization & Activations

This section covers all arguments related to normalization layers and activation functions.

## Normalization

These arguments control the normalization layers used throughout the model. Normalization is crucial for stabilizing training and improving the model's performance.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--norm_variant_attn` | str | "rmsnorm" | The normalization variant to use for the attention layers. Options include "krmsnorm", "prmsnorm", "rmsnorm", "layernorm", "hyperspherenorm", "dact", "identity". |
| `--norm_variant_output` | str | "rmsnorm" | The normalization variant to use for the final output of the Transformer blocks. |
| `--use_pre_ln` | bool | True | If `True`, applies normalization before the attention and MLP layers (standard practice in modern LLMs). |
| `--use_post_ln` | bool | False | If `True`, applies normalization after the attention and MLP layers (the original Transformer architecture). |
| `--use_peri_ln` | bool | False | If `True`, applies normalization both before and after the attention and MLP layers. |

<br>

---

## Normalization Variant Specific Arguments

<details>
<summary><b>PRMSNorm</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--prmsnorm_pct` | float | 0.0625 | The percentage of the initial entries to use for the partial RMS calculation. This is a more efficient variant of RMSNorm. |
</details>

<details>
<summary><b>KRMSNorm</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--krmsnorm_num` | int | 10 | The maximum number of initial entries to use for the partial RMS calculation. |
| `--krmsnorm_quantize_type` | str | "none" | The quantization type to use for KRMSNorm. Options are "int8", "int16", "none". |
| `--krmsnorm_enable_gain` | bool | True | Whether to include a learnable gain parameter in KRMSNorm. |
| `--krmsnorm_selection_type` | str | "last" | The method for selecting which entries to use. Options are "first", "last", "random". |
| `--krmsnorm_recompute_percentage` | float | None | The percentage of the total RMS that must be within the partial RMS to avoid recomputation. |
</details>

<details>
<summary><b>HyperSphereNorm</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--hsnorm_gain` | bool | False | Whether to include a learnable gain parameter. |
| `--hsnorm_radius` | float | None | The radius of the hypersphere. |
| `--hsnorm_radius_learning` | bool | False | Whether the radius is a learnable parameter. |
</details>

<details>
<summary><b>DynamicActivations (dact)</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dact_activation` | str | "tanh" | The activation function to use within the DynamicActivations module. |
| `--dact_use_gamma` | bool | True | Whether to use a learnable gain (gamma). |
| `--dact_use_beta` | bool | True | Whether to use a learnable bias (beta). |
| `--dact_alpha_init` | float | 1.0 | The initial value for the alpha parameter. |
| `--dact_use_alpha` | bool | True | Whether to use a learnable alpha parameter. |
</details>

<br>

---

## Activation Functions

These arguments control the activation functions used in the MLP layers.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--activation_variant` | str | "gelu" | The activation function to use. Options include "celu", "elu", "gelu", "gelu_shifted", "glu", "leaky_relu", "learned_spline", "mish", "piecewise", "pfla", "pfla_le", "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid", "silu", "softplus", "softsign", "squared_relu", "tanh", "identity". |

<br>

---

## Activation Variant Specific Arguments

<details>
<summary><b>Shifted GeLU</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--shifted_gelu_learnable_shift` | bool | True | Whether the shift parameter is learnable. |
| `--shifted_gelu_initial_shift` | float | 0.0 | The initial value for the shift. |
</details>

<details>
<summary><b>PiecewiseLearnableActivation (pla)</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pla_num_points` | int | 7 | The number of points in the piecewise function. |
| `--pla_left_bound` | float | -2.0 | The left bound of the function's domain. |
| `--pla_right_bound` | float | 2.0 | The right bound of the function's domain. |
</details>

<details>
<summary><b>PiecewiseFullyLearnableActivation (pfla)</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pfla_num_points` | int | 50 | The number of points in the piecewise function. |
| `--pfla_left_bound` | float | -10.0 | The left bound of the function's domain. |
| `--pfla_right_bound` | float | 10.0 | The right bound of the function's domain. |
</details>

<details>
<summary><b>LearnedSplineActivation (lsa)</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--lsa_num_knots` | int | 30 | The number of knots in the spline. |
</details>
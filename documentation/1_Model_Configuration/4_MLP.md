# 1.4 MLP Layers

This section provides a comprehensive guide to all arguments that control the Multi-Layer Perceptron (MLP) within the Transformer blocks. The MLP is the primary non-linear component of the model and is responsible for much of its representational power.

## Core MLP Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mlp_variant` | str | "mlp" | The main MLP architecture to use. Options include "mlp" (a standard feed-forward network), "kan" (Kolmogorov-Arnold Networks), "swiglu" (SwiGLU activation), "dual_path", and "identity" (which effectively removes the MLP). |
| `--mlp_expansion_factor` | int | 4 | The expansion factor for the hidden layer of the MLP. The hidden layer size will be `mlp_expansion_factor * n_embd`. A value of 4 is standard. |
| `--mlp_size` | int | None | If specified, this value is used as the exact size of the MLP's hidden layer, overriding `--mlp_expansion_factor`. |
| `--mlp_down_projs` | int | 1 | The number of down-projection layers in the MLP/SwiGLU. |
| `--use_parallel_mlp` | bool | False | If `True`, the MLP layer is applied in parallel with the attention layer, rather than sequentially. This is an experimental feature. |

## Bias Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mlp_up_bias` | bool | None | Whether to use a bias in the up-projection layer of the MLP. If `None`, the global `--bias` setting is used. |
| `--mlp_down_bias` | bool | None | Whether to use a bias in the down-projection layer of the MLP. If `None`, the global `--bias` setting is used. |

## Offset Configuration

These arguments allow for adding learnable or fixed offsets to the MLP's activation function. This can be useful for shifting the activation function's operating range.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mlp_x_offset` | float | 0.0 | A fixed offset to add to the input of the MLP's activation function. |
| `--mlp_y_offset` | float | 0.0 | A fixed offset to add to the output of the MLP's activation function. |
| `--learn_mlp_x_offset` | bool | False | If `True`, the x-axis offset is a learnable parameter. |
| `--learn_mlp_y_offset` | bool | False | If `True`, the y-axis offset is a learnable parameter. |

<br>

---

## MLP Variant Specific Arguments

<details>
<summary><b>KAN (Kolmogorov-Arnold Networks)</b></summary>

These arguments are specific to the KAN (Kolmogorov-Arnold Networks) MLP variant.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--kan_poly_order` | int | 3 | The polynomial order for the KAN non-linearity. |
| `--kan_base_activation` | str | "silu" | The base activation function for the KAN. |
| `--kan_middle_layers` | list | [] | A list of integers specifying the sizes of the middle layers in the KAN. |
</details>
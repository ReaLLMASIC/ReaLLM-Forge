# 1.7 Quantization

This section provides a comprehensive guide to all arguments related to the hardware-aware quantization features of the model. Quantization is the process of reducing the precision of the model's weights and activations, which can lead to significant reductions in memory usage and inference speed, with a minimal impact on accuracy.

## Global Quantization Scheduler

These arguments control the global quantization level, which can be scheduled to change during training. This allows for a gradual introduction of quantization, which can help maintain model accuracy.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--start_quant_level` | float | 0.0 | The starting level of quantization. A value of 0.0 means no quantization, and 1.0 means full quantization. |
| `--full_quant_iteration` | int | None | The training iteration at which the model should reach full quantization. |
| `--quant_scheduler` | str | None | The scheduler for changing the quantization level. "static" keeps the quantization level constant, while "linear" increases it linearly from `start_quant_level` to 1.0 over `full_quant_iteration` steps. |
| `--quantization_warmup_iters` | int | 100 | The number of iterations to use regular linear layers before switching to quantized linear layers. This can help stabilize training. |

## Embedding Quantization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_wte` | bool | None | Whether to quantize the word token embeddings. |
| `--quantize_wte_method` | str | "affine_quant" | The quantization method for `wte`. Options are "ternary_quant", "symmetric_quant", "affine_quant", "stochastic_quant". |
| `--quantize_wte_bits` | int | 8 | The number of bits for `wte` quantization. |
| `--quantize_wpe` | bool | None | Whether to quantize the word position embeddings. |
| `--quantize_wpe_method` | str | "affine_quant" | The quantization method for `wpe`. |
| `--quantize_wpe_bits` | int | 8 | The number of bits for `wpe` quantization. |

## Activation Quantization

These arguments control the quantization of activations within the Transformer blocks.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--activations_quant_method` | str | "affine_quant" | The default quantization method for activations. |
| `--store_activations` | bool | False | Whether to store the activations as a buffer and update them throughout training. This is necessary for some quantization methods. |

<details>
<summary><b>Attention Activation Quantization</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_attn_act` | bool | False | A global switch to quantize all input/output activations in the attention layers. |
| `--quantize_attn_act_bits` | int | 8 | The default number of bits for attention activation quantization. |
| `--quantize_attn_act_input` | bool | False | Quantizes the input activation to the attention layer. |
| `--quantize_attn_act_input_bits` | int | None | Overrides the default bits for the attention input. |
| `--quantize_attn_act_qk_mult_q_input` | bool | False | Quantizes the query input to the QK multiplication. |
| `--quantize_attn_act_qk_mult_q_input_bits`| int | None | Overrides the default bits for the QK multiplication query input. |
| ... | ... | ... | ... |
</details>

<details>
<summary><b>MLP Activation Quantization</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_mlp_act` | bool | False | A global switch to quantize all input/output activations in the MLP layers. |
| `--quantize_mlp_act_bits` | int | 8 | The default number of bits for MLP activation quantization. |
| `--quantize_mlp_act_input` | bool | False | Quantizes the input activation to the MLP layer. |
| `--quantize_mlp_act_input_bits` | int | None | Overrides the default bits for the MLP input. |
| ... | ... | ... | ... |
</details>

## Linear Layer Quantization

These arguments control the quantization of the weights of the linear layers.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--linear_variant_attn` | str | "linear" | The linear layer variant for the attention layers. Set to "quantized_linear" to enable quantization. |
| `--linear_variant_mlp` | str | "linear" | The linear layer variant for the MLP layers. Set to "quantized_linear" to enable quantization. |
| `--quantize_linear_method` | str | "affine_quant" | The default quantization method for linear layers. |
| `--quantize_linear_bits` | int | 8 | The default number of bits for linear layer quantization. |

<details>
<summary><b>Granular Linear Quantization</b></summary>

These arguments allow you to specify different quantization methods and bitwidths for each linear layer in the model.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_linear_attn_q_method` | str | None | Overrides the default method for the query projection. |
| `--quantize_linear_attn_q_bits` | int | None | Overrides the default bits for the query projection. |
| ... | ... | ... | ... |
</details>
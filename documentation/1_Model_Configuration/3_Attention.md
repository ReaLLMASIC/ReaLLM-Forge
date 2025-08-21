# 1.3 Attention Mechanisms

This section provides a comprehensive guide to all arguments that control the attention mechanism within the Transformer blocks.

## Core Attention Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--attention_variant` | str | "causal" | The main attention mechanism to use. Options include "causal" (standard self-attention), "linear", "ssm" (Structured State Space Model), and more experimental variants. |
| `--attention_list` | list | None | A list of attention variants to cycle through for each layer. This allows for creating models with mixed attention types (e.g., `causal causal ssm causal causal ssm`). |
| `--disable_flash_attention` | bool | False | Manually disables the use of Flash Attention, forcing a fallback to the slower, manual attention implementation. This is useful for debugging or when Flash Attention is not supported. |
| `--use_qk_norm` | bool | False | If `True`, applies normalization (typically RMSNorm) to the query (Q) and key (K) vectors before the attention score calculation. This can help stabilize training. |
| `--use_qk_norm_scale` | bool | False | If `True` and using `--use_qk_norm`, applies a learnable scaling factor to the QK norm. |
| `--attn_logit_softcapping` | bool | None | If specified, applies a softcapping function (like `tanh`) to the attention logits before masking and softmax. This can prevent extreme values and stabilize training. |

## Grouped Query Attention (GQA)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_kv_group` | int | None | The number of key-value (KV) groups. If specified and less than `n_head`, enables Grouped Query Attention (GQA). The number of Q heads will be `n_head`, while the number of K and V heads will be `n_kv_group`. `n_head` must be divisible by `n_kv_group`. GQA is a technique to reduce the computational cost of attention by sharing K and V heads among multiple Q heads. |

<br>

---

## Attention Variant Specific Arguments

<details>
<summary><b>Flash LoBo & OBO</b></summary>

These arguments control a specific variant of Flash Attention.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_flash_lobo` | bool | False | Enables the LoBo (Logarithmic Bi-linear Operator) variant of Flash Attention. |
| `--use_flash_lobo_per_head` | bool | False | If `True`, each attention head will have its own LoBo parameters. |
| `--use_flash_obo_const` | bool | False | If `True`, makes the Flash LoBo parameter a non-learnable constant. |
| `--flash_lobo_log_const` | float | 0.0 | The initial value for the LoBo log constant. |
</details>

<details>
<summary><b>MLA (Multi-Layer Attention)</b></summary>

These arguments control the MLA (Multi-Layer Attention) variant.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mla_latent_dim` | int | None | The projected latent size for MLA. Defaults to `n_embd // 4`. |
| `--mla_rotary_dim` | int | 32 | The dimensionality of the rotary branch used by MLA. |
| `--use_mla_lobo` | bool | False | Enables the LoBo variant for MLA. |
| `--mla_lobo_init` | float | 0.0 | The initial value for the MLA LoBo log constant. |
</details>

<details>
<summary><b>CO4 (Co-occurrence Attention)</b></summary>

These arguments control the CO4 (Co-occurrence Attention) variant.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_latent` | int | 4 | The number of latent queries (Lq). |
| `--triadic_loops` | int | 1 | The number of Q-K-V refinement passes. |
| `--mod_fn` | str | "cooperation" | The MOD transfer-function to use. Options are "cooperation", "tm1", "tm2", "tm3", "tm4". |
</details>

<details>
<summary><b>Infinite Attention</b></summary>

These arguments control the Infinite Attention variant.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_qk_head_dim` | int | None | The dimensionality of the query and key heads. |
| `--n_v_head_dim` | int | None | The dimensionality of the value heads. |
| `--n_cproj` | int | None | The dimensionality of the output projection. |
| `--use_concat_heads` | bool | False | If `True`, concatenates the heads instead of adding them in the output projection. |
</details>

<details>
<summary><b>SSM (Structured State Space Model)</b></summary>

These arguments control the Structured State Space Model (SSM) variant, which is based on the Mamba architecture.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ssm_mamba_expand` | int | 2 | The expansion factor for the Mamba block. |
| `--ssm_conv_kernel_size` | int | 3 | The kernel size for the convolutional layer in the Mamba block. |
| `--ssm_dt_rank` | int | 8 | The rank for the discretization step. |
| `--ssm_d_state` | int | 16 | The dimensionality of the state vector. |
| `--ssm_io_bias` | bool | False | Whether to add biases to the input and output projections. |
</details>
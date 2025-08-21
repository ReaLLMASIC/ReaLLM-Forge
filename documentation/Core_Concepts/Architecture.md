# 🏛️ Model Architecture

The core of the ReaLLM-Forge repository is a highly modular and configurable GPT-style model, defined in `model.py`. This document details the architecture and its various components, which can be configured via the `GPTConfig` object (defined in `gpt_conf.py`) and command-line arguments in `train_args.py`.

## Core Architecture

These arguments control the fundamental size and shape of the Transformer model.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_layer` | int | 6 | The number of Transformer blocks in the model. |
| `--n_head` | int | 6 | The number of attention heads in each Transformer block. |
| `--n_embd` | int | 384 | The dimensionality of the token and position embeddings. |
| `--block_size` | int | 256 | The maximum sequence length that the model can process. |
| `--dropout` | float | 0.0 | The dropout rate applied to the model's layers. |
| `--bias` | bool | False | Whether to use bias in the model's linear and normalization layers. |

---

## Embeddings

### Word Token Embeddings

These arguments control the word token embedding layer (`wte`).

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_embd_wte` | int | None | If specified, creates a factorized embedding table with this dimensionality. An adapter table will be automatically created to project the embeddings to `n_embd`. |
| `--n_embd_wte_scale_tying` | bool | True | If using factorized embeddings, whether to tie the weights of the scale-up and scale-down matrices. |
| `--wte_weight_tying` | bool | True | Whether to tie the weights of the word token embedding layer and the final language model head. |
| `--init_variant` | str | "gaussian" | The initialization scheme for the embedding weights. Options include "gaussian", "onehot", "hypercube", etc. |
| `--embedding_mean_init` | float | 0.0 | The mean of the normal distribution for Gaussian initialization of embedding weights. |
| `--embedding_std_init` | float | 0.02 | The standard deviation of the normal distribution for Gaussian initialization of embedding weights. |

### Positional Embeddings

These arguments control the positional embedding scheme.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_abs_pos_embeddings` | bool | True | Whether to use standard absolute positional embeddings. |
| `--use_rotary_embeddings` | bool | False | Whether to use Rotary Positional Embeddings (RoPE). |
| `--rope_variant` | str | "rope" | The variant of RoPE to use. Options are "rope" and "soap". |
| `--rope_length` | int | None | The number of embeddings to apply RoPE to. Must be an even number. |
| `--use_fire_embeddings` | bool | False | Whether to use Functional Interpolation for Relative Positional Encoding (FIRE). |
| `--shared_fire_embeddings` | bool | False | Whether to share the FIRE embeddings across layers. |

---

## Attention Mechanisms

### Attention Variants

These arguments control the core attention mechanism used in the Transformer blocks.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--attention_variant` | str | "causal" | The attention mechanism to use. Options include "causal", "linear", "ssm", "identity", "infinite", "mla", "co4". |
| `--attention_list` | list | None | A list of attention variants to cycle through for each layer. |
| `--disable_flash_attention` | bool | False | Whether to disable Flash Attention. |
| `--use_qk_norm` | bool | False | Whether to apply normalization to the query and key vectors before the attention calculation. |

### Grouped Query Attention (GQA)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_kv_group` | int | None | The number of key-value groups for Grouped Query Attention. If not specified, standard multi-head attention is used. |

---

## MLP Layers

### MLP Variants

These arguments control the structure of the multi-layer perceptron (MLP) in the Transformer blocks.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mlp_variant` | str | "mlp" | The MLP architecture to use. Options include "mlp", "kan", "swiglu", "dual_path", "identity". |
| `--mlp_expansion_factor` | int | 4 | The expansion factor for the hidden layer of the MLP. |
| `--mlp_size` | int | None | If specified, overrides `mlp_expansion_factor` to set the exact size of the MLP's hidden layer. |
| `--mlp_up_bias`, `--mlp_down_bias` | bool | None | Whether to use bias in the up and down projection layers of the MLP. If `None`, the global `--bias` setting is used. |

---

## Normalization Layers

These arguments control the normalization layers used in the model.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--norm_variant_attn` | str | "rmsnorm" | The normalization variant to use for the attention layers. |
| `--norm_variant_output` | str | "rmsnorm" | The normalization variant to use for the final output of the Transformer blocks. |
| `--use_pre_ln` | bool | True | Whether to apply normalization before the attention and MLP layers. |
| `--use_post_ln` | bool | False | Whether to apply normalization after the attention and MLP layers. |
| `--use_peri_ln` | bool | False | Whether to apply normalization both before and after the attention and MLP layers. |

---

## Activation Functions

These arguments control the activation functions used in the MLP layers.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--activation_variant` | str | "gelu" | The activation function to use. Options include "gelu", "relu", "silu", "tanh", etc. |

---

## Softmax Functions

These arguments control the softmax function used in the attention mechanism and the final output layer.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--softmax_variant_attn` | str | "softmax" | The softmax variant to use for the attention layers. |
| `--softmax_variant_output` | str | "softmax" | The softmax variant to use for the final output layer. |

---

## Quantization

These arguments control the quantization of the model's weights and activations.

### Embedding Quantization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_wte` | bool | None | Whether to quantize the word token embeddings. |
| `--quantize_wte_method` | str | "affine_quant" | The quantization method for `wte`. |
| `--quantize_wte_bits` | int | 8 | The number of bits for `wte` quantization. |
| `--quantize_wpe` | bool | None | Whether to quantize the word position embeddings. |
| `--quantize_wpe_method` | str | "affine_quant" | The quantization method for `wpe`. |
| `--quantize_wpe_bits` | int | 8 | The number of bits for `wpe` quantization. |

### Activation Quantization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_attn_act` | bool | False | Whether to quantize all activations in the attention layers. |
| `--quantize_mlp_act` | bool | False | Whether to quantize all activations in the MLP layers. |
| ... | ... | ... | ... |

### Linear Quantization

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantize_linear_method` | str | "affine_quant" | The quantization method for linear layers. |
| `--quantize_linear_bits` | int | 8 | The number of bits for linear layer quantization. |
| ... | ... | ... | ... |

---

## Advanced Features

### Mixture of Experts (MoE)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_moe` | bool | False | Whether to use a Mixture of Experts architecture. |
| `--moe_layer_freq` | int | 2 | The frequency at which to replace FFNs with MoE layers. |
| `--n_experts` | int | 8 | The number of experts per MoE layer. |
| `--moe_top_k` | int | 2 | The number of experts to route to for each token. |

### Steering Vectors

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_lsv` | bool | False | Whether to use Learned Steering Vectors. |
| `--apply_vector_at_layer_idx` | int | None | The layer index at which to apply a steering vector. |
| ... | ... | ... | ... |
# 1.2 Embeddings

This section covers all arguments related to the model's embedding layers, which are responsible for converting token IDs and positions into vector representations.

## Word Token Embeddings (WTE)

These arguments control the main embedding layer that maps vocabulary indices to vectors.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_embd_wte` | int | None | If specified, creates a factorized embedding table with this smaller dimensionality. An adapter (two linear layers) will be automatically created to project these smaller embeddings up to the model's main dimension (`n_embd`) and back down to the LM head. This is a form of parameter reduction that can significantly reduce memory usage, especially for large vocabularies. |
| `--n_embd_wte_scale_tying` | bool | True | If using factorized embeddings (`--n_embd_wte`), this ties the weights of the scale-up and scale-down projection matrices, further reducing the parameter count. |
| `--wte_weight_tying` | bool | True | Ties the weights of the word token embedding layer and the final language model head. This is a common and effective practice for reducing parameters and can improve performance by coupling the input and output representations. |
| `--import_wte_npy` | str | None | Path to a `.npy` file to import a pre-trained embedding table. This is useful for transfer learning or initializing with specific embeddings. |
| `--export_wte_npy` | str | None | Path to save the model's embedding table to a `.npy` file after training. |
| `--export_wte_each_eval` | bool | False | If `True`, the embedding table will be exported at every evaluation interval, allowing you to observe how embeddings change during training. |
| `--import_wte_freeze` | bool | False | If `True`, the imported embedding table will be frozen and not updated during training. |
| `--import_scale_matrices_npz` | str | None | Path to a `.npz` file to import pre-trained scale-up and scale-down matrices for factorized embeddings. |
| `--export_scale_matrices_npz` | str | None | Path to save the scale-up and scale-down matrices to a `.npz` file. |
| `--export_scale_matrices_each_eval` | bool | False | If `True`, the scale matrices will be exported at every evaluation interval. |
| `--import_scale_matrices_freeze` | bool | False | If `True`, the imported scale matrices will be frozen. |

## Positional Embeddings

These arguments control the method used to encode positional information into the model. You can use absolute and relative positional embeddings simultaneously.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_abs_pos_embeddings` | bool | True | Whether to use standard, learned absolute positional embeddings. This is the traditional method where each position in the sequence has a unique, learnable vector. |
| `--use_rotary_embeddings` | bool | False | Whether to use Rotary Positional Embeddings (RoPE), a relative positional encoding scheme that is applied to queries and keys in the attention mechanism. |
| `--rope_variant` | str | "rope" | The variant of RoPE to use. Options are "rope" and "soap". |
| `--rope_length` | int | None | The number of embeddings to apply RoPE to. Must be an even number. If `None`, it applies to all embeddings. |
| `--use_fire_embeddings` | bool | False | Whether to use Functional Interpolation for Relative Positional Encoding (FIRE), another relative positional encoding scheme. |
| `--shared_fire_embeddings` | bool | False | Whether to share the FIRE embeddings across all layers. |

## Initialization

These arguments control the initialization scheme for the embedding weights.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--init_variant` | str | "gaussian" | The initialization scheme for the embedding weights. Options include "gaussian", "onehot", "hypercube", "numpy_import", "rand_hypercube", "angle_hypersphere", "unique_hypercube", "gaussian_norm_range". |
| `--init_scale` | float | 0.01 | The scaling factor for non-Gaussian initialization schemes. |
| `--init_wte_npy` | str | "wte.npy" | The `.npy` file to use for the "numpy_import" initialization scheme. |
| `--init_radius` | float | 1.0 | The radius for the "angle_hypersphere" initialization scheme. |
| `--gaussian_min_norm` | float | 0.0 | The minimum norm for the "gaussian_norm_range" initialization scheme. |
| `--gaussian_max_norm` | float | inf | The maximum norm for the "gaussian_norm_range" initialization scheme. |
| `--embedding_mean_init` | float | 0.0 | The mean of the normal distribution for Gaussian initialization of embedding weights. |
| `--embedding_std_init` | float | 0.02 | The standard deviation of the normal distribution for Gaussian initialization of embedding weights. |
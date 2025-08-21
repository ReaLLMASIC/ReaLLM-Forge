# 1.10 Other Advanced Features

This section covers other advanced and experimental features that can be configured for the model.

## Shared Parameters

These arguments allow for sharing parameters between different layers of the model, which can reduce the total parameter count and potentially improve performance. This is a form of parameter tying.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--shared_mlp_size` | int | 1 | The number of contiguous blocks that share the same MLP weights. A value of 1 means no sharing. |
| `--shared_mlp_sym` | bool | False | If `True`, enables symmetrical sharing of MLP weights. |
| `--shared_mlp_seq` | int | 1 | The sequence length for cyclic sharing of MLP layers. |
| `--shared_attn_size` | int | 1 | The number of contiguous blocks that share the same attention weights. |
| `--shared_attn_sym` | bool | False | If `True`, enables symmetrical sharing of attention weights. |
| `--shared_attn_seq` | int | 1 | The sequence length for cyclic sharing of attention layers. |

## Gradient Checkpointing

This feature can reduce the memory usage of the model during training, at the cost of increased computation time. It works by not storing the intermediate activations of the forward pass, and instead recomputing them during the backward pass.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_gradient_checkpointing` | bool | False | A global switch to enable gradient checkpointing. |
| `--recompute_backward_pass` | bool | False | If `True`, recomputes the backward pass, which is necessary for gradient checkpointing. |

## Learned Position Embeddings (LPE)

These arguments control the use of Learned Position Embeddings, which are a separate set of Transformer blocks that learn position-aware residuals. This is an alternative to the standard positional embedding schemes.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_lpe` | int | 0 | The number of LPE modules to instantiate. |
| `--lpe_block_size` | int | 256 | The block size for the LPE modules. |
| `--lpe_n_layer` | int | 3 | The number of layers for the LPE modules. |
| `--lpe_n_head` | int | 6 | The number of attention heads for the LPE modules. |
| `--target_layer_in_lpe` | int | 0 | The layer at which to input the activations to the LPE modules. |
| `--target_layer_out_lpe` | int | 0 | The layer at which to add the output of the LPE modules. |

## Learned Confidence Residual Scaling

These arguments control the use of a learned confidence scaling mechanism for the residual connections. This allows the model to learn how much to trust the output of each residual block.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_attn_resid_scaling` | bool | False | Whether to apply learned confidence scaling to the attention outputs. |
| `--use_mlp_resid_scaling` | bool | False | Whether to apply learned confidence scaling to the MLP outputs. |
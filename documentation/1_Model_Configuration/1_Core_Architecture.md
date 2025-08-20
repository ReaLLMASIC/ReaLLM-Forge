# 1.1 Core Architecture

These arguments control the fundamental size and shape of the Transformer model. They are the most critical parameters for defining the model's capacity and performance.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n_layer` | int | 6 | The total number of Transformer blocks (or layers) in the model. Increasing this value makes the model "deeper," allowing it to learn more complex hierarchical features. |
| `--n_head` | int | 6 | The number of attention heads in each Transformer block. A higher number of heads allows the model to jointly attend to information from different representational subspaces at different positions. `n_embd` must be divisible by `n_head`. |
| `--n_embd` | int | 384 | The dimensionality of the token and position embeddings, and consequently the internal hidden state of the model. This is a primary lever for controlling the model's size. |
| `--block_size` | int | 256 | The maximum sequence length (context window) that the model can process at one time. Longer sequences require more memory and compute. |
| `--dropout` | float | 0.0 | The dropout rate applied to various layers for regularization. It is a technique to prevent overfitting by randomly setting a fraction of neuron activations to zero during training. A value of 0.0 means no dropout. |
| `--bias` | bool | False | A global switch to enable or disable bias parameters in all `Linear` and `LayerNorm` layers throughout the model. Disabling bias can sometimes improve performance and speed, and is a common practice in modern LLMs. |
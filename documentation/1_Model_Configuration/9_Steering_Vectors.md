# 1.9 Steering Vectors

This section provides a comprehensive guide to all arguments related to both manual and Learned Steering Vectors (LSVs). Steering vectors are a technique for controlling the behavior of a model by adding a vector to its activations at a specific layer.

## Manual Steering Vectors

These arguments allow you to apply a pre-computed steering vector to the model's activations at a specific layer. This is useful for tasks like style transfer or controlling the sentiment of the generated text.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--apply_vector_at_layer_idx` | int | None | The layer index at which to apply the steering vector. |
| `--apply_vector_file` | str | None | The `.npy` file containing the steering vector to apply. |
| `--apply_vector_scaling_factor` | float | 1.0 | The scaling factor to apply to the steering vector before adding it to the activations. |
| `--obtain_vector_at_layer_idx` | int | None | The layer index from which to obtain an activation vector. This is useful for creating new steering vectors. |
| `--obtain_vector_file` | str | None | The `.npy` file to save the obtained activation vector to. |

## Learned Steering Vectors (LSVs)

These arguments control the use of Learned Steering Vectors, which are trained to steer the model's behavior based on the current dataset. This is a more advanced feature that allows the model to learn how to control its own behavior.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_lsv` | bool | False | A global switch to enable the use of Learned Steering Vectors. |
| `--lsv_index` | int | None | The index of the steering vector to use. |
| `--lsv_variant` | str | "one_hot" | The variant of LSV to use. Options include "one_hot", "linear_comb", "one_hot_mlp", etc. |
| `--apply_lsv_at_layer_idx` | int | None | The layer index at which to apply the LSV. |
| `--lsv_focused_training` | bool | False | If `True`, only the LSV parameters will be trained, while the rest of the model is frozen. This is useful for fine-tuning the steering vectors. |
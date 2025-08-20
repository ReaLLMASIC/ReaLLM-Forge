# 3.4 Model Exporting

This section provides a comprehensive guide to all arguments related to the exporting of the model and its components.

## ONNX Export

ONNX (Open Neural Network Exchange) is a standard format for representing machine learning models. Exporting the model to ONNX can be useful for deploying it to a variety of different platforms and frameworks.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--onnx_output` | bool | False | If `True`, the model will be exported to the ONNX format. |

## Embedding Export

These arguments allow you to export the model's embedding tables to NumPy files. This can be useful for analyzing the embeddings or for using them in other applications.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--export_wte_npy` | str | None | The path to export the word token embedding table to a `.npy` file. |
| `--export_wte_each_eval` | bool | False | If `True`, the embedding table will be exported at every evaluation interval. |
| `--export_scale_matrices_npz` | str | None | The path to export the scale matrices for factorized embeddings to a `.npz` file. |
| `--export_scale_matrices_each_eval` | bool | False | If `True`, the scale matrices will be exported at every evaluation interval. |

## Quantization Data Export

| Argument | Type | Default | Description |
|---|---|---|---|
| `--quantization_data_file` | str | None | If specified, the quantized weights, activations, scale factors, and zero points will be exported to this file. This is useful for hardware-aware training and for deploying quantized models. |
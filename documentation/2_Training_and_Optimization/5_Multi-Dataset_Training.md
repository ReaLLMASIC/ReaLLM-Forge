# 2.5 Multi-Dataset Training

This section provides a comprehensive guide to all arguments related to the advanced multi-dataset and multi-context training modes.

## Core Multi-Dataset Configuration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--training_mode` | str | 'single' | The training mode. Set to 'multidataset' or 'multicontext' to enable multi-dataset training. |
| `--dataset_list` | list | None | A list of datasets to use for 'multidataset' training. |
| `--dataset_sampling_probs` | list | None | The probabilities for sampling from each dataset in 'multidataset' mode. |
| `--dataset_interleaving` | bool | False | If `True`, interleaves the datasets instead of sampling from them. This can be useful for ensuring that the model sees data from all datasets in a more regular fashion. |
| `--dataset_interleaving_shuffle` | bool | False | If `True`, shuffles the interleaved datasets. |
| `--dataset_sampling_learning_rate` | list | None | A list of learning rates to use for each dataset. This allows you to use different learning rates for different datasets, which can be useful if the datasets have different characteristics. |
| `--multicontext_datasets` | list | None | A list of datasets to use for 'multicontext' training. In this mode, the model processes batches from multiple datasets in parallel. |
| `--multidataset_wte` | bool | False | If `True`, uses separate token embeddings and language model heads for each dataset. This is useful if the datasets have different vocabularies. |
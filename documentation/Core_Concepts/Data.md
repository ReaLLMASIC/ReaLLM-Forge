# 📊 Data Pipeline

This document explains how to prepare, manage, and use datasets with the ReaLLM-Forge training scripts.

## Data Format

The training script (`train.py`) expects a specific data format. Each dataset should have its own directory inside the `data/` folder. This directory must contain the following files:

-   `train.bin`: A binary file containing the tokenized training data as a flat array of token IDs.
-   `val.bin`: A binary file containing the tokenized validation data, in the same format as `train.bin`.
-   `meta.pkl`: A Python pickle file containing metadata about the dataset. At a minimum, this file must contain a dictionary with the following keys:
    -   `vocab_size`: An integer representing the total number of tokens in the vocabulary.
    -   `stoi`: A dictionary mapping tokens (characters or subwords) to their corresponding integer IDs.
    -   `itos`: A dictionary mapping integer IDs back to their corresponding tokens.

## Preparing a New Dataset

To prepare a new dataset, you can use the provided script `data/create_new_dataset.sh`. This script will create a new directory in `data/` with a template structure.

1.  **Create the Dataset Directory:**
    ```bash
    bash data/create_new_dataset.sh <your_dataset_name>
    ```

2.  **Add Your Data:** Place your raw text data (e.g., `input.txt`) into the newly created `data/<your_dataset_name>` directory.

3.  **Implement the Tokenizer:** The script will create a `prepare.py` file in your dataset directory. You will need to edit this file to implement your desired tokenization scheme. The `prepare.py` script should:
    -   Read your raw text data.
    -   Create the `stoi` and `itos` mappings.
    -   Tokenize the text into integer IDs.
    -   Split the data into training and validation sets.
    -   Save the tokenized data to `train.bin` and `val.bin`.
    -   Save the metadata to `meta.pkl`.

4.  **Run the Preparation Script:**
    ```bash
    python data/<your_dataset_name>/prepare.py
    ```

## Using Existing Datasets

Many common datasets are already included in the `data/` directory. To use one of these, you simply need to run the provided `get_dataset.sh` script within the dataset's directory. For example, to download and prepare the Shakespeare dataset:

```bash
bash data/shakespeare_char/get_dataset.sh
```

Once the data is prepared, you can use it in your training runs with the `--dataset` argument:

```bash
python train.py --dataset=shakespeare_char
```

## Data-Related Training Arguments

These arguments from `train_args.py` control how data is handled during training.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | 'shakespeare_char' | The name of the dataset directory in `data/`. |
| `--training_mode` | str | 'single' | The training mode. Options are 'single', 'multidataset', and 'multicontext'. |
| `--dataset_list` | list | None | A list of datasets to use for 'multidataset' training. |
| `--dataset_sampling_probs` | list | None | The probabilities for sampling from each dataset in 'multidataset' mode. |
| `--multicontext_datasets` | list | None | A list of datasets to use for 'multicontext' training. |
| `--batch_size` | int | 64 | The number of sequences to process in parallel. |
| `--sampling_method` | str | "random" | The method for sampling batches from the dataset. Options are "random", "sequential", and "without_replacement". |
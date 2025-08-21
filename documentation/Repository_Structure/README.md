# 📂 Repository Structure

This section provides a detailed overview of the directory structure of the ReaLLM-Forge repository.

## Root Directory

| File/Folder | Description |
|---|---|
| [`analysis/`](./Folders/analysis.md) | Contains scripts for analyzing model checkpoints, activations, and other aspects of the training process. |
| [`benchmarks/`](./Folders/benchmarks.md) | Contains scripts for running benchmarks and evaluating model performance. |
| [`colabs/`](./Folders/colabs.md) | Contains Jupyter notebooks for exploring the repository and its features. |
| [`data/`](./Folders/data.md) | Contains datasets and scripts for preparing them for training. |
| [`demos/`](./Folders/demos.md) | Contains example shell scripts for running various training and evaluation workflows. |
| [`distillation/`](./Folders/distillation.md) | Contains scripts for model distillation. |
| [`documentation/`](./Folders/documentation.md) | Contains the comprehensive documentation for the repository. |
| [`explorations/`](./Folders/explorations.md) | Contains configuration files for experiments and hyperparameter sweeps. |
| [`exutorch/`](./Folders/exutorch.md) | Contains the ExuTorch library. |
| [`hp_searches/`](./Folders/hp_searches.md) | Contains logs and configurations for hyperparameter searches. |
| [`huggingface_model/`](./Folders/huggingface_model.md) | Contains scripts for converting models to the Hugging Face format. |
| [`HW/`](./Folders/HW.md) | Contains hardware-related files and scripts. |
| [`initializations/`](./Folders/initializations.md) | Contains scripts for custom weight initializations. |
| [`logging/`](./Folders/logging.md) | Contains logging-related files and scripts. |
| [`optimization_and_search/`](./Folders/optimization_and_search.md) | Contains scripts for running automated experiments and hyperparameter searches. |
| [`quantization/`](./Folders/quantization.md) | Contains scripts and modules related to model quantization. |
| [`report/`](./Folders/report.md) | Contains files for generating reports. |
| [`tests/`](./Folders/tests.md) | Contains tests for the repository's code. |
| [`tf_np_golden_gen/`](./Folders/tf_np_golden_gen.md) | Contains scripts for generating golden data from TensorFlow and NumPy. |
| [`train_variations/`](./Folders/train_variations.md) | Contains scripts for different training variations. |
| [`util_factorization/`](./Folders/util_factorization.md) | Contains utility scripts for factorization. |
| [`utils/`](./Folders/utils.md) | Contains utility scripts and modules used throughout the repository. |
| [`variations/`](./Folders/variations.md) | Contains the different architectural variations for the model's components. |
| `model.py` | The core GPT model architecture. |
| `train.py` | The main script for training models. |
| `sample.py` | The script for generating text (inference) from a trained model. |
| `gpt_conf.py` | The `GPTConfig` dataclass, which holds all hyperparameters for a given model and experiment. |
| `hyperparam_search.py` | A script for performing a greedy, iterative hyperparameter search. |
| ... | ... |
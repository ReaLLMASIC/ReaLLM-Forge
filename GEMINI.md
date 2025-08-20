# Gemini Agent Onboarding: nanogpt_gemini_md

This document provides a guide for AI agents to understand and interact with this repository.

## 1. Project Overview

This is a repository for training and experimenting with GPT-style language models, forked from nanoGPT. Its primary focus is on hardware-aware training and extensive experimentation with different model architectures, tokenization methods, and datasets to analyze their impact on Power, Performance, and Area (PPA) for hardware implementation.

The framework is highly modular, allowing researchers to easily swap components (e.g., attention mechanisms, normalization layers, activation functions) and configure experiments through JSON or YAML files.

## 2. Directory Structure

-   **/data:** Contains scripts and subdirectories for datasets. New datasets can be added using the `create_new_dataset.sh` script, and each subdirectory typically has a script to download and prepare the data.
-   **/demos:** Example shell scripts demonstrating various training and evaluation workflows.
-   **/explorations:** Configuration files (`.json`, `.yaml`) for different experiments. These files are used by the training scripts to define model architecture and hyperparameters.
-   **/hp_searches:** Contains logs and configurations for hyperparameter searches.
-   **/analysis:** Contains tools for inspecting results after training, such as analyzing checkpoints (`checkpoint_analysis`) or model activations (`activation_analysis`).
-   **/model.py:** Defines the core GPT model architecture. It is highly configurable via the `GPTConfig` object.
-   **/train.py:** The main script for training models. It is highly parameterized to allow for a wide range of experiments.
-   **/sample.py:** The script for generating text (inference) from a trained model. It includes advanced features for visualization and analysis.
-   **/gpt_conf.py:** Defines the `GPTConfig` dataclass, which holds all hyperparameters for a given model and experiment.
-   **/hyperparam_search.py:** A script for performing a greedy, iterative hyperparameter search.

## 3. Key Scripts and Usage

### `train.py`
-   **Purpose:** The main script for training a new model or fine-tuning an existing one.
-   **Key Arguments:**
    -   `--dataset`: Specifies the dataset to use from the `data/` directory.
    -   `--n_layer`, `--n_head`, `--n_embd`: Define the model architecture.
    -   `--batch_size`, `--learning_rate`: Control training parameters.
    -   `--init_from`: Can be 'scratch', 'resume', or a GPT-2 variant (e.g., 'gpt2').
    -   `--out_dir`: Directory to save checkpoints and logs.
    -   `--compile`: Compiles the model with `torch.compile` for a speed-up.
    -   Many other arguments defined in `train_args.py` and reflected in `gpt_conf.py` to control architectural variations.

### `sample.py`
-   **Purpose:** Generates samples from a trained model checkpoint.
-   **Key Arguments:**
    -   `--out_dir`: Path to the checkpoint directory to load the model from.
    -   `--start`: The initial prompt for generation (can be a string or `FILE:prompt.txt`).
    -   `--num_samples`: How many samples to generate.
    -   `--max_new_tokens`: The maximum length of the generated text.
    -   `--temperature`, `--top_k`: Control the randomness and diversity of the output.
    -   `--colorize_output`: A boolean flag that enables color-coding the output tokens based on the model's confidence.
    -   `--show_heatmaps`: Generates visualizations of token probabilities at each step.

### `hyperparam_search.py`
-   **Purpose:** Performs a greedy search over a specified hyperparameter space. It iteratively adjusts parameters from a baseline configuration, runs training, and selects the change that provides the best efficiency (e.g., score improvement per added parameter).
-   **Key Arguments:**
    -   `--orig_settings`: A YAML file with the baseline configuration (e.g., `baseline.yaml`).
    -   `--param_names`: A list of parameters to search over (e.g., `n_layer n_head`).
    -   `--increments`: The step size to adjust each parameter by.
    -   `--iterations`: How many steps to take for each parameter in each search direction.
    -   `--num_iterations`: The total number of greedy search iterations to perform.
    -   `--results_file`: The YAML file to log the results of the sweep.

## 4. Common Workflows

### Workflow A: Training a Single Model
1.  **Prepare Data:** Ensure your dataset is located in `data/<your_dataset_name>` and has been preprocessed. For a new dataset, you might use `data/create_new_dataset.sh`. For existing ones, run the `get_dataset.sh` script inside the specific data directory (e.g., `bash data/shakespeare_char/get_dataset.sh`).
2.  **Configure:** Use command-line arguments to specify the model and training parameters.
3.  **Execute Training:** Run the training script. A simple example is:
    ```bash
    python train.py --dataset=shakespeare_char --batch_size=64 --max_iters=5000 --out_dir=out/my_shakespeare_run
    ```

### Workflow B: Running an Automated Experiment Sweep
This workflow is useful for running a set of predefined training runs from a configuration file.
1.  **Define Experiment:** Create or modify a JSON/YAML configuration file in the `explorations/` directory. This file contains a list of training configurations to run.
2.  **Execute Sweep:** Use the `optimization_and_search/run_experiments.py` script to execute the sweep.
    ```bash
    python optimization_and_search/run_experiments.py -c explorations/config.json
    ```

### Workflow C: Performing a Hyperparameter Search
This workflow automatically searches for optimal hyperparameters.
1.  **Define Baseline:** Create a `baseline.yaml` file that specifies the starting configuration for the search.
2.  **Configure Search:** Create a shell script (like `run_hp_search.sh`) to define the search space using the arguments for `hyperparam_search.py`.
3.  **Execute Search:** Run the script.
    ```bash
    bash run_hp_search.sh
    ```
4.  **Monitor Results:** Check the output YAML file (e.g., `random_lhem_out.yaml`) to see the results of each trial and the best-performing configurations.

### Workflow D: Running Inference
1.  **Identify Checkpoint:** Locate the `ckpt.pt` file in the output directory from your training run (e.g., `out/my_shakespeare_run`).
2.  **Execute Sampling:** Run the sampling script, pointing it to the correct output directory.
    ```bash
    python sample.py --out_dir=out/my_shakespeare_run --start="ROMEO: "
    ```

## 5. Advanced Usage

### Activating Architectural Variations
The model's architecture can be significantly altered via command-line flags passed to `train.py`. These correspond to settings in `gpt_conf.py`. Key variations include:
-   **Attention:** `--attention_variant` (e.g., `causal`, `ssm`), `--use_flash_lobo`
-   **MLP:** `--mlp_variant` (e.g., `mlp`, `kan`, `swiglu`)
-   **Normalization:** `--norm_variant_attn`, `--norm_variant_output` (e.g., `rmsnorm`, `layernorm`)
-   **Activation Function:** `--activation_variant` (e.g., `gelu`, `relu`, `silu`)
-   **Softmax:** `--softmax_variant_attn`, `--softmax_variant_output` (e.g., `softmax`, `consmax`)
-   **Positional Embeddings:** `--use_rotary_embeddings`, `--use_fire_embeddings`

### Data Preparation Details
The training script expects a specific data format. Each dataset directory inside `data/` should contain:
-   `train.bin`: A binary file containing the tokenized training data.
-   `val.bin`: A binary file containing the tokenized validation data.
-   `meta.pkl`: A pickle file containing metadata, including the vocabulary size (`vocab_size`) and the tokenizer information (`stoi`, `itos` for character-level, or tokenizer name). The `data/template` directory provides a starting point.

### Benchmarking and Analysis
-   **Standardized Benchmarking:** To evaluate a trained model on standard benchmarks, use the `--lm_eval_tasks` flag in `sample.py`. This will run the specified tasks from the `lm-evaluation-harness`.
    ```bash
    python sample.py --out_dir=out/my_run --lm_eval_tasks="hellaswag,arc_easy"
    ```
-   **Monitoring Training:** During or after a training run, you can inspect the checkpoints to find the best validation loss using `analysis/checkpoint_analysis/inspect_ckpts.py`.
    ```bash
    python analysis/checkpoint_analysis/inspect_ckpts.py --directory ./out
    ```

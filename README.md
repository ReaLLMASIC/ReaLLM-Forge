# ReaLLM-Forge: A Framework for Hardware-Aware LLM Exploration

ReaLLM-Forge is a highly modular and extensible framework for training and experimenting with GPT-style language models, forked from nanoGPT. It is designed to facilitate research into hardware-aware training, novel architectures, and the impact of various hyperparameters on model performance.

This repository bridges the gap between theoretical model design and practical hardware implementation, ensuring efficient, scalable, and robust ML model development. It provides a rich set of tools for exploring a wide range of model configurations, from different attention mechanisms and MLP structures to advanced features like quantization and Mixture of Experts.

For a comprehensive guide to the repository's features and workflows, please see the [**Full Documentation**](./documentation/README.md).

## Key Features

-   **Extensive Architectural Variations:** Easily experiment with different attention mechanisms, MLP layers, normalization schemes, and activation functions.
-   **Hardware-Aware Training:** Explore features like quantization and embedding factorization to understand their impact on hardware performance.
-   **Automated Experiment Management:** Run systematic experiment sweeps and hyperparameter searches with the provided scripts.
-   **Advanced Visualization and Analysis:** Use the integrated tools to visualize model outputs, analyze performance, and gain insights into your models.
-   **Flexible Data Pipeline:** Easily add new datasets and tokenization schemes.

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ReaLLM-Forge.git
    cd ReaLLM-Forge
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Quick Start: Training a Model

1.  **Prepare the data:**
    ```bash
    bash data/shakespeare_char/get_dataset.sh
    ```

2.  **Start training:**
    ```bash
    python train.py --dataset=shakespeare_char --out_dir=out/my_shakespeare_run
    ```

3.  **Generate text:**
    ```bash
    python sample.py --out_dir=out/my_shakespeare_run --start="ROMEO: "
    ```

## Citation

This work extends Andrej Karpathy's foundational [nanoGPT](https://github.com/karpathy/nanoGPT). We ask that citations cite both projects.

To cite **ReaLLM-Forge**, please use this BibTeX entry:

```bibtex
@software{ReaLLM-Forge,
  author = {{ReaLLMASIC} and {Contributors}},
  title = {{ReaLLM-Forge: A Framework for Hardware-Aware LLM Exploration}},
  url = {https://github.com/ReaLLMASIC/ReaLLM-Forge},
  year = {2025},
}
```
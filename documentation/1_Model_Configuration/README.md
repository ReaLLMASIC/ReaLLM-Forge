# ⚙️ 1. Model Configuration

This section provides a complete and detailed reference for every argument that controls the model's architecture, as defined in `train_args.py`. The model's structure is highly configurable, allowing for extensive experimentation with different components and variations.

## Table of Contents

### Core Architecture & Embeddings
-   [**Core Architecture**](./1_Core_Architecture.md): Arguments that define the fundamental size and shape of the Transformer model.
-   [**Embeddings**](./2_Embeddings.md): Configuration for all embedding layers, including token, positional, and advanced schemes like RoPE and FIRE.

### Major Components & Variations
-   [**Attention Mechanisms**](./3_Attention.md): A deep dive into all attention mechanism variations, from standard causal attention to GQA, Flash Attention, and experimental variants.
-   [**MLP Layers**](./4_MLP.md): A complete guide to the different Multi-Layer Perceptron (MLP) architectures, including KAN and SwiGLU.
-   [**Normalization & Activations**](./5_Normalization_and_Activations.md): Documentation for all normalization schemes (RMSNorm, LayerNorm, etc.) and activation functions (GeLU, ReLU, etc.).
-   [**Softmax Functions**](./6_Softmax.md): A detailed reference for all available softmax variations and their specific tuning parameters.

### Advanced & Experimental Features
-   [**Quantization**](./7_Quantization.md): A comprehensive guide to the hardware-aware quantization features for weights, activations, and embeddings.
-   [**Mixture of Experts (MoE)**](./8_Mixture_of_Experts.md): Configuration for the Mixture of Experts architecture.
-   [**Steering Vectors**](./9_Steering_Vectors.md): Documentation for using both manual and Learned Steering Vectors (LSVs).
-   [**Other Advanced Features**](./10_Other_Advanced_Features.md): A reference for other advanced and experimental features like shared parameters and gradient checkpointing.

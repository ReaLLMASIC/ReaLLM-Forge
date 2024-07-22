from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_group: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    window_size: int = 128
    gate: bool = False
    quantization_linear_method: str = "affine_quant"
    quantization_linear_bits: int = 8
    quantize_wte: bool = False
    quantize_wpe: bool = False
    quantization_embedding_method: str = "affine_quant"
    quantization_embedding_bits: int = 8
    quantize_attn_all: bool = False
    quantize_c_attn_q: bool = False
    quantize_c_attn_k: bool = False
    quantize_c_attn_v: bool = False
    quantize_q: bool = False
    quantization_q_bits: int = 8
    quantize_k: bool = False
    quantization_k_bits: int = 8
    quantize_v: bool = False
    quantization_v_bits: int = 8
    quantize_q_k_mult: bool = False
    quantize_softmax_v_mult: bool = False
    quantization_mult_bits: int = 8
    quantize_softmax: bool = False
    quantization_softmax_bits: int = 8
    quantize_activation: bool = False
    quantization_activation_bits: int = 8
    quantization_activation_method: str = "affine_quant"
    quantize_attn_proj: bool = False
    quantization_attn_proj_bits: int = 8
    quantize_mlp_all: bool = False
    quantize_mlp_up: bool = False
    quantize_mlp_down: bool = False

    # Training options
    ## Gradient Checkpointing - More memory efficient (can do long contexts), but is slower
    use_gradient_checkpointing: bool = False

    # MLP Options
    use_parallel_mlp: bool = False
    use_swiglu: bool = False

    # Shared parameters
    # MLP
    shared_mlp_size: int = 1
    shared_mlp_sym: bool = False
    # ATTN
    shared_attn_size: int = 1
    shared_attn_sym: bool = False

    # Softmax Alternatives and Options
    softmax_variant_attn: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"
    softmax_variant_output: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"

    ## General Options
    div_by_seq_len: bool = False # for supported functions will divide by seq length

    ## ConSmax Options
    consmax_initial_beta: float = 2.0 # beta adjustment
    consmax_initial_gamma: float = 100.0 # denominator adjustment
    consmax_use_euler_base: bool = True # use 'e' as base for ConSmax, default
    consmax_base: float = 2.0 # base to utilize for ConSmax

    ## SaturatingConSmax special options (otherwise same as ConSmax)
    consmax_saturation: float = 11.0 # for SaturatingConSmax saturation point
    consmax_learnable_beta: bool = True
    consmax_learnable_gamma: bool = True

    ## Softermax options
    softermax_use_xmax: bool = True # Softermax Option active is softermax selected - True: uses (x - x_max) normalization; False: removes normalization (potential overflow)

    ## Polymax options
    polymax_x_intercept: float = -100.0
    polymax_y_intercept: float = 1.0
    polymax_power: float = 2.0
    polymax_divisor: float = 1000.0

    ## SigSoftmaxBase
    sigsoftmax_use_euler_base: bool = True # use 'e' as base for Constantmax
    sigsoftmax_base: float = 2.0 # denominator to utilize for Constantmax

    ## Strongermax options
    strongermax_strength: float = 2.0 # Softermax with option of 'stronger' (larger integer) bases
    strongermax_sum_to_1: bool = False # Softermax with option of 'stronger' (larger integer) bases
    strongermax_divisor: float = 1.0 # Softermax with option of 'stronger' (larger integer) bases
    strongermax_use_xmax: bool = True # Softermax with option of 'stronger' (larger integer) bases

    ## ExpPolymax options
    exppolymax_use_euler_base: bool = True
    exppolymax_base: float = 2.719
    exppolymax_y_intercept: float = 1.0
    exppolymax_power: float = 2.0
    exppolymax_divisor: float = 1.0

    ## Softplus options
    softplus_divisor: float = 100.0

    ## Squareplus options
    squareplus_divisor: float = 100.0

    # Positional Embeddings Variations
    use_abs_pos_embeddings: bool = True # Note: one can use this AND rotary embeddings
    use_fire_embeddings: bool = False
    shared_fire_embeddings: bool = False
    use_rotary_embeddings: bool = False
    sym_rot_num_angles: int = 512
    rope_variant: str = "rope" # options: "shortrope", "rope"
    shortrope_length: int = 8 # number of embeddings to use in shortrope

    # Structuring Options, remember to compile the model
    use_post_ln: bool = True

    # Layernorm Alternatives and Options
    norm_variant_attn: str = "rmsnorm"
    norm_variant_output: str = "rmsnorm"
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    prmsnorm_pct: float = 0.0625
    krmsnorm_num: float = 10

    # Activation Alternatives
    activation_variant: str = "gelu"

    # Linear Alternatives
    attn_linear_variant: str = "linear"
    mlp_linear_variant: str = "linear"

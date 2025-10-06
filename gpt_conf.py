# gpt_conf.py
from dataclasses import dataclass, field, asdict, fields
from typing import List
import json
import math

@dataclass
class GPTConfig:
    attention_list: List[str] = field(default_factory=lambda: [])
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_group: int = 12
    n_embd: int = 768
    mlp_down_projs: int = 1  # Number of down projections in MLP/SwiGLU

    # numerical multicontext
    numerical_multicontext: bool = False
    numerical_mlp_hidden_dim: int = 64

    # Layerlists
    n_head_layerlist: List[int] = field(default_factory=list)
    n_qk_head_dim_layerlist: List[int] = field(default_factory=list)
    n_v_head_dim_layerlist: List[int] = field(default_factory=list)
    mlp_size_layerlist: List[int] = field(default_factory=list)

    # For multicontext training
    multicontext: bool = False
    # Use separate embeddings/LM heads per dataset in multidataset mode
    multidataset_wte: bool = False
    vocab_sizes: List[int] = field(default_factory=lambda: []) # Used in place of vocab

    # MLP bias configuration
    mlp_up_bias: bool | None = None  # If None, uses global bias setting
    mlp_down_bias: bool | None = None  # If None, uses global bias setting

    # FFN offset parameters
    mlp_x_offset: float = 0.0
    mlp_y_offset: float = 0.0
    learn_mlp_x_offset: bool = False
    learn_mlp_y_offset: bool = False

    # Optional L2 normalization of MLP projections
    l2_norm_mlp_up: bool = False
    l2_norm_mlp_down: bool = False
    l2_norm_mlp_up_dim: str = "embed"   # 'embed' or 'hidden'
    l2_norm_mlp_down_dim: str = "hidden"  # 'embed' or 'hidden'

    ## MLA Variations
    mla_latent_dim: int | None = None   # d_c  (proj dimension of the shared latent)
    mla_rotary_dim: int       = 32      # d_r  (# rotary channels per head)

    use_mla_lobo: bool = False          # turns the feature on/off
    mla_lobo_init: float = 0.0          # log-space initial value (like flash_lobo_log_const)

    ## CO4 attention variation
    n_latent: int = None
    triadic_loops: int = None
    mod_fn: str = "causal"

    ## Inf attention variation
    n_qk_head_dim: int = None
    n_v_head_dim: int = None
    n_cproj: int = None
    use_concat_heads: bool = False

    # Softcapping params
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None

    # Final ln_f input mixing
    use_ln_f_input_mixer: bool = False
    ln_f_input_mixer_variant: str = "linear"
    ln_f_mixer_top_k: int = 2

    # Learned Position Embeddings
    n_lpe: int = 0
    lpe_block_size: int = 1024
    lpe_n_layer: int = 12
    lpe_n_head: int = 12
    lpe_n_kv_group: int = 12
    lpe_n_qk_head_dim: int = None
    lpe_n_v_head_dim: int = None
    lpe_use_abs_pos_embeddings: bool = True
    lpe_use_rotary_embeddings: bool = True
    lpe_attention_variant: str = "causal"
    lpe_mlp_variant: str = "mlp"
    lpe_mlp_size: str = None
    target_layer_in_lpe: int = 0
    target_layer_out_lpe: int = 0

    # Shared parameters
    # MLP
    lpe_shared_mlp_size: int = 1
    lpe_shared_mlp_sym: bool = False
    # ATTN
    lpe_shared_attn_size: int = 1
    lpe_shared_attn_sym: bool = False

    # Attention Variation Specific

    ## Flash Lobo
    use_flash_lobo: bool = False
    use_flash_lobo_per_head: bool = False
    use_flash_obo_const: bool = False
    flash_lobo_log_const: float = 0.0

    # Steering Vectors
    ## Where to intercept
    apply_vector_at_layer_idx: int = None
    obtain_vector_at_layer_idx: int = None
    use_lsv: bool = False
    lsv_index: int = None
    lsv_dataset_num: int = None
    lsv_variant: str = "one_hot"
    apply_lsv_at_layer_idx: int = None

    ## Files to insert or obtain vectors from
    apply_vector_file: str = None
    apply_vector_scaling_factor: float = 1.0
    obtain_vector_file: str = None

    # If Factorizing:
    n_embd_wte: int = None

    # weight tying
    n_embd_wte_scale_tying: bool = True
    wte_weight_tying: bool = True # Non-factorized wte weight tying

    # wte import/export
    import_wte_freeze: bool = False
    import_wte_npy: str = None
    export_wte_npy: str = None
    export_wte_each_eval: bool = False

    # scaling matrices import/export
    import_scale_matrices_freeze: bool = False
    import_scale_matrices_npz: str = None
    export_scale_matrices_npz: str = None
    export_scale_matrices_each_eval: bool = False

    dropout: float = 0.0
    window_size: int = None
    use_flex_attn: bool = None

    gate: bool = False
    use_moe: bool = False
    moe_layer_freq: int = 2
    n_experts: int = 8
    moe_top_k: int = 2
    moe_router_scheme: str = "softmax"

    # Logging options
    softmax_io_logging: bool = False
    consmax_beta_gamma_logging: bool = False
    plot_statistics: bool = False
    softmax_io_log_interval: int = 1

    # Training options
    ## Gradient Checkpointing - More memory efficient (can do long contexts), but is slower
    use_gradient_checkpointing: bool = False
    recompute_backward_pass: bool = False

    ## Flash attention
    disable_flash_attention: bool = False

    # Attention Options
    attention_variant: str = "causal"

    # QK Norm Options
    use_qk_norm: bool = False
    use_qk_norm_scale: bool = False
    use_v_norm: bool = False

    ## SSM - Attention Varient (same as Hymba)
    ssm_mamba_expand: int = 2
    ssm_conv_kernel_size: int = 3
    ssm_dt_rank: int = 8
    ssm_d_state: int = 16
    ssm_io_bias: bool = True

    # EdgeLLM ASIC block architecture
    use_edgellm_asic: bool = False

    # MLP Options
    use_parallel_mlp: bool = False
    mlp_variant: str = "mlp"
    mlp_expansion_factor: int = 4
    mlp_size: int = None
    mlp_cproj_scale: float = 1.0
    mlp_post_act_l2_norm: bool = False

    ## KAN Option
    kan_poly_order: int = 3
    kan_base_activation: str = "silu"
    kan_middle_layers: List[int] = field(default_factory=lambda: [])

    # Shared parameters
    # MLP
    shared_mlp_size: int = 1
    shared_mlp_sym: bool = False
    shared_mlp_seq: int = 1
    # ATTN
    shared_attn_size: int = 1
    shared_attn_sym: bool = False
    shared_attn_seq: int = 1

    # Softmax Alternatives and Options
    softmax_variant_attn: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"
    softmax_variant_output: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"


    ## General Options
    div_by_seq_len: bool = False # for supported functions will divide by seq length

    ## ConSmax Options
    consmax_initial_beta: float = 2.0 # beta adjustment
    consmax_initial_gamma: float = 100.0 # denominator adjustment
    consmax_base: float = 2.0 # base to utilize for ConSmax
    consmax_use_euler_base: bool = True # use 'e' as base for ConSmax, default

    ## ConSmaxV2 Special Options
    consmax_per_head: bool = True # different beta gamma per head
    consmax_v2_clamping: bool = True
    consmax_v2_clamp_value: float = 80.0

    ## SaturatingConSmax Special options (otherwise same as ConSmax)
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
    strongermax_strength: float = math.e
    strongermax_div_by_sum_of_terms: bool = True
    strongermax_divisor: float = 1.0

    strongermax_use_xmax: bool = True

    strongermax_xmax_guess: float = 1.0
    strongermax_overflow_recompute: bool = False
    strongermax_overflow_recompute_value: float = 88.0

    strongermax_clamping: bool = False
    strongermax_clamp_value: float = 88.0

    strongermax_obo: float = 0.0
    strongermax_use_learned_obo: bool = False
    strongermax_use_learned_obo_per_head: bool = False

    strongermax_temperature_factor: float = 1.0
    strongermax_use_learned_temperature_factor: bool = False

    ## ExpPolymax options
    exppolymax_use_euler_base: bool = True
    exppolymax_base: float = math.e
    exppolymax_y_intercept: float = 1.0
    exppolymax_power: float = 2.0
    exppolymax_divisor: float = 1.0

    ## Softplus options
    softplus_divisor: float = 256.0

    ## ReLUMax options
    relumax_divisor: float = 256.0

    ## ReLUMax options
    relu2max_divisor: float = 256.0

    ## SigmoidMax options
    sigmoidmax_divisor: float = 256.0

    ## SoftShrink options
    softshrink_attn_lambda: float = 0.5
    softshrink_attn_divisor: float = 64.0

    ## Squareplus options
    squareplus_divisor: float = 256.0

    # ──────────────────────────────────────────────────────────────────────
    ## PFLA‑Softmax configuration
    pfla_softmax_num_points: int  = 30      # # inner control points
    pfla_softmax_left_bound: float  = -10.0 # x‑range start
    pfla_softmax_right_bound: float = 10.0  # x‑range end
    pfla_softmax_learn_x: bool  = False     # learn knot x‑positions?
    pfla_softmax_learn_y: bool  = True      # learn √y values?
    pfla_softmax_init_activation: str = "gelu"   # reference curve for init
    pfla_softmax_density: str = "linear"    # linear | quad | exp

    ### normalisation extras
    pfla_softmax_use_learned_divisor: bool = False
    pfla_softmax_gamma_init: float  = 1.0   # γ initial value iff learned

    pfla_softmax_use_obo: bool = False # enables obo feature
    pfla_softmax_use_learned_obo: bool = False # we require "use_obo" before use_learned_obo
    pfla_softmax_obo: float = 0.0           # fixed or initial OBO (+1) addend

    ### interpolation mode
    pfla_softmax_mode: str = "linear"       # "linear" | "quadratic"
    # ──────────────────────────────────────────────────────────────────────

    # Positional Embeddings Variations
    use_abs_pos_embeddings: bool = True # Note: one can use this AND rotary embeddings
    use_fire_embeddings: bool = False
    shared_fire_embeddings: bool = False
    use_rotary_embeddings: bool = False
    sym_rot_num_angles: int = 512
    rope_variant: str = "rope" # options: "shortrope", "rope"
    rope_length: int = 8 # number of embeddings to use in shortrope

    ## Embedding Intialization Options
    embedding_mean_init: float= 0.0
    embedding_std_init: float= 0.02

    ## FIRE Options (Functional Interpolation for Relative Positional Encoding)
    fire_log_bias: float = 1.0
    fire_num_hidden_layers: int = 1
    fire_mlp_width: int = 32
    fire_init_c: float = 0.1
    fire_init_L: float = 512.0
    fire_outermost_sigma: bool = False

    # Structuring Options, remember to compile the model
    use_post_ln: bool = False
    use_pre_ln: bool = True
    use_peri_ln: bool = False
    use_pre_ln_attn: bool | None = None
    use_pre_ln_mlp: bool | None = None
    use_peri_ln_attn: bool | None = None
    use_peri_ln_mlp: bool | None = None
    use_post_ln_attn: bool | None = None
    use_post_ln_mlp: bool | None = None
    use_attn_resid_scaling: bool = False
    use_mlp_resid_scaling: bool = False
    attn_confidence_variant: str = "zeros"
    mlp_confidence_variant: str = "zeros"
    use_attn_resid_const: bool = False
    attn_resid_const: float = 0.0
    learn_attn_resid_const: bool = False
    use_mlp_resid_const: bool = False
    mlp_resid_const: float = 0.0
    learn_mlp_resid_const: bool = False
    resid_gaussian_mean_init: float = 0.0
    resid_gaussian_std_init: float = 0.02
    attn_residual_combination: str = "add"
    mlp_residual_combination: str = "add"
    residual_slerp_eps: float = 0.0
    attn_residual_alpha: float = 0.05
    mlp_residual_alpha: float = 0.05
    attn_residual_alpha_type: str = "fixed"
    mlp_residual_alpha_type: str = "fixed"

    # Layernorm Alternatives and Options
    norm_variant_attn: str = "rmsnorm"
    norm_variant_output: str = "rmsnorm"
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    prmsnorm_pct: float = 0.0625
    krmsnorm_num: float = 10
    krmsnorm_quantize_type: str = 'int8'
    krmsnorm_enable_gain: bool = True
    krmsnorm_selection_type: str = 'last'
    krmsnorm_recompute_percentage: float = 0.05
    hsnorm_gain: bool = False
    hsnorm_radius: float = 1.0
    hsnorm_radius_learning: bool = False

    dact_alpha_init: float = 1.0
    dact_activation: str = 'tanh'
    dact_use_gamma: bool = True
    dact_use_beta: bool = True
    dact_use_alpha: bool = True
    use_embedding_scale: bool = False
    l2_norm_embed: bool = False
    l2_norm_embed_scale: float | None = None
    l2_norm_embed_scale_learnable: bool = True

    l2_norm_lm_head: bool = False
    l2_norm_lm_head_scale: float | None = None
    l2_norm_lm_head_scale_learnable: bool = True

    # Activation Alternatives

    activation_variant: str = "gelu"

    ## Shifted Gelu
    shifted_gelu_learnable_shift: bool = True
    shifted_gelu_initial_shift: float = 0.0

    ## Softshrink
    softshrink_lambda: float = 0.5

    ## PiecewiseLearnableActivation - pla
    pla_num_points: int = 7
    pla_left_bound: float = -2.0
    pla_right_bound: float = 2.0

    ## PiecewiseFullyLearnableActivation - pfla
    pfla_num_points: int = 200
    pfla_left_bound: float = -100.0
    pfla_right_bound: float = 100.0

    ## LearnedSplineActivation - lsa
    lsa_num_knots: int = 30


    # Linear Alternatives
    linear_variant_attn: str = "linear"
    linear_variant_mlp: str = "linear"
    linear_variant_q: str = None
    linear_variant_k: str = None
    linear_variant_v: str = None
    linear_variant_attn_proj: str = None
    linear_variant_mlp_up: str = None
    linear_variant_mlp_down: str = None

    ## Linear Initialization Options
    linear_mean_init: float= 0.0
    linear_std_init: float= 0.02

    ## Embedding initialization options
    init_variant: str = "gaussian"
    init_scale: float = 0.01
    init_wte_npy: str = "wte.npy"
    init_radius: float = 1.0
    gaussian_min_norm: float = 0.0
    gaussian_max_norm: float = float('inf')

    # Quantizations
    start_quant_level: float = 0
    quant_scheduler: str = None
    full_quant_iteration: int = None
    # Needed for quant_level printing
    eval_interval: int = 250

    ## Embedding Quantizations
    quantize_wte: bool = False
    quantize_wpe: bool = False
    quantize_wte_method: str = "affine_quant"
    quantize_wte_bits: int = 8
    quantize_wpe_method: str = "affine_quant"
    quantize_wpe_bits: int = 8

    ## Activation Quantizations
    activations_quant_method: str = "affine_quant"
    quantize_attn_act: bool = False
    quantize_attn_act_bits: int = 8
    quantize_attn_act_input: bool = False
    quantize_attn_act_input_bits: int = None
    quantize_attn_act_qk_mult_q_input: bool = False
    quantize_attn_act_qk_mult_q_input_bits: int = None
    quantize_attn_act_qk_mult_k_input: bool = False
    quantize_attn_act_qk_mult_k_input_bits: int = None
    quantize_attn_act_softmax_input: bool = False
    quantize_attn_act_softmax_input_bits: int = None
    quantize_attn_act_pv_mult_p_input: bool = False
    quantize_attn_act_pv_mult_p_input_bits: int = None
    quantize_attn_act_pv_mult_v_input: bool = False
    quantize_attn_act_pv_mult_v_input_bits: int = None
    quantize_attn_act_pv_mult_output: bool = False
    quantize_attn_act_pv_mult_output_bits: int = None
    quantize_attn_act_output: bool = False
    quantize_attn_act_output_bits: int = None
    quantize_mlp_act: bool = False
    quantize_mlp_act_bits: int = 8
    quantize_mlp_act_input: bool = False
    quantize_mlp_act_input_bits: int = None
    quantize_mlp_act_activation_input: bool = False
    quantize_mlp_act_activation_input_bits: int = None
    quantize_mlp_act_activation_output: bool = False
    quantize_mlp_act_activation_output_bits: int = None
    quantize_mlp_act_output: bool = False
    quantize_mlp_act_output_bits: int = None
    quantize_asic_prenorm: bool = False
    quantize_asic_offchip_residual: bool = False
    quantize_asic_bits: int = None
    quantize_asic_attn_softmax_denom: bool = False
    quantize_asic_attn_softmax_denom_bits: int = None
    quantize_asic_attn_softmax_numerator: bool = False
    quantize_asic_attn_softmax_numerator_bits: int = None
    store_activations: bool = False

    ## Linear Quantizations
    quantize_linear_method: str = "affine_quant"
    quantize_linear_bits: int = 8
    quantize_linear_attn_q_method: str = None
    quantize_linear_attn_q_bits: int = None
    quantize_linear_attn_k_method: str = None
    quantize_linear_attn_k_bits: int = None
    quantize_linear_attn_v_method: str = None
    quantize_linear_attn_v_bits: int = None
    quantize_linear_attn_proj_method: str = None
    quantize_linear_attn_proj_bits: int = None
    quantize_linear_mlp_up_method: str = None
    quantize_linear_mlp_up_bits: int = None
    quantize_linear_mlp_down_method: str = None
    quantize_linear_mlp_down_bits: int = None
    quantization_warmup_iters: int = 100

    @classmethod
    def from_json(cls, filename: str):
        try:
            with open(filename, 'r') as json_file:
                config_dict = json.load(json_file)

            # Get all field names of the dataclass
            field_names = {f.name for f in fields(cls)}

            # Filter the loaded dict to only include valid fields
            filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

            # Create and return a new instance
            return cls(**filtered_dict)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: File '{filename}' is not a valid JSON file.")
            return None
        except TypeError as e:
            print(f"Error: Invalid data in JSON file. {str(e)}")
            return None

    def to_json(self, filename: str):
        """
        Function to save a GPTConfig object as json to be used for later model creation

        input:
        - fout: string = filename of saved config file

        """
        conf_dict = asdict(self)

        with open(filename, 'w') as json_file:
            json.dump(conf_dict, json_file)


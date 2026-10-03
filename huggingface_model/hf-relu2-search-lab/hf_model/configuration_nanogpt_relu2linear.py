"""Configuration for the Hugging Face compatible nanoGPT model."""

import math

from transformers import PretrainedConfig

from .relu2linear_activation import validate_threshold


class NanoGPTReLU2LinearConfig(PretrainedConfig):
    """Separate experimental architecture; original NanoGPTConfig is unchanged.

    Set mode/threshold BEFORE constructing the model. Attention modules snapshot
    them so mutating the config mid-training cannot change recomputation math.
    Both fields are serialized by save_pretrained. Existing divisor/accelerator
    field names are retained to keep other experimental settings identical.
    """

    model_type = "nanogpt_relu2linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        max_position_embeddings=1024,
        hidden_size=768,
        intermediate_size=None,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=None,
        hidden_act="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
        tie_word_embeddings=True,
        use_qk_norm=True,
        qk_norm_eps=1e-6,
        use_qk_norm_scale=True,
        qk_norm_scale_init=None,
        use_rotary_embeddings=True,
        rope_theta=10000.0,
        rope_length=None,
        use_absolute_position_embeddings=False,
        attention_normalizer="relu2linear",
        relu2linear_mode="fixed8",
        relu2linear_threshold=8.0,
        relu2max_divisor=256.0,
        relu2max_divide_by_sequence_length=False,
        relu2max_accelerator="auto",
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        intermediate_size = intermediate_size or 4 * hidden_size
        num_key_value_heads = num_key_value_heads or num_attention_heads
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
        rope_length = head_dim if rope_length is None else rope_length
        if rope_length < 0 or rope_length > head_dim or rope_length % 2:
            raise ValueError("rope_length must be even and in [0, head_dim]")
        if attention_normalizer != "relu2linear":
            raise ValueError("Use NanoGPTConfig for the unchanged softmax/ReLU2Max models")
        if relu2linear_mode not in {"fixed8", "configurable"}:
            raise ValueError("relu2linear_mode must be 'fixed8' or 'configurable'")
        relu2linear_threshold = validate_threshold(relu2linear_threshold)
        if relu2linear_mode == "fixed8" and relu2linear_threshold != 8.0:
            raise ValueError("fixed8 requires threshold=8; select configurable for another threshold")
        self.relu2linear_mode = relu2linear_mode
        self.relu2linear_threshold = relu2linear_threshold
        if not isinstance(relu2max_divisor, (int, float)) or isinstance(relu2max_divisor, bool) or not math.isfinite(relu2max_divisor) or relu2max_divisor <= 0:
            raise ValueError("relu2max_divisor must be positive")
        if relu2max_accelerator not in {"auto", "torch", "triton"}:
            raise ValueError("relu2max_accelerator must be 'auto', 'torch', or 'triton'")
        if qk_norm_scale_init is None:
            qk_norm_scale_init = math.log2(
                max_position_embeddings * max_position_embeddings
                - max_position_embeddings
            )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.use_qk_norm_scale = use_qk_norm_scale
        self.qk_norm_scale_init = qk_norm_scale_init
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rope_theta = rope_theta
        self.rope_length = rope_length
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.attention_normalizer = attention_normalizer
        self.relu2max_divisor = relu2max_divisor
        self.relu2max_divide_by_sequence_length = relu2max_divide_by_sequence_length
        self.relu2max_accelerator = relu2max_accelerator
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

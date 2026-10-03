"""Opt-in configuration for the matched full-model experiment."""
import math
from .configuration_nanogpt import NanoGPTConfig
from .relu2linear_activation import validate_threshold
from .context_attention import CONTEXT_ARMS, validate_length_scaling

VARIANTS = ("softmax_sdpa", "relu2", "linear2", "linear4", "linear8", "linear16", "linear_config8", "linear_config", "kda", *CONTEXT_ARMS)
FIXED_THRESHOLDS = {"linear2": 2.0, "linear4": 4.0, "linear8": 8.0, "linear16": 16.0, "linear_config8": 8.0}


class ComparisonConfig(NanoGPTConfig):
    model_type = "nanogpt_relu2_comparison"

    def __init__(self, variant="relu2", linear_threshold=6.5, sdpa_backend="flash",
                 require_fused=True, norm_eps=1e-6, output_norm="none", kda_conv_size=4,
                 subtractive_bias_init=0.0, causal_length_anchor=512.0, **kwargs):
        uses_bias, alpha = CONTEXT_ARMS.get(variant, (False, 0.0))
        validate_length_scaling(alpha, causal_length_anchor)
        if not isinstance(subtractive_bias_init, (int, float)) or isinstance(subtractive_bias_init, bool) or not math.isfinite(subtractive_bias_init):
            raise ValueError("subtractive_bias_init must be finite")
        self.subtractive_bias_init = float(subtractive_bias_init)
        self.causal_length_anchor = float(causal_length_anchor)
        # Persist and validate the treatment, so changing just a checkpoint's
        # variant label cannot silently alter its attention rule on reload.
        for key, value in (("learned_subtractive_bias", uses_bias), ("causal_length_alpha", alpha)):
            if key in kwargs and kwargs.pop(key) != value:
                raise ValueError(f"{key} disagrees with variant {variant}")
            setattr(self, key, value)
        if variant in CONTEXT_ARMS and kwargs.get("relu2max_divide_by_sequence_length", False):
            raise ValueError("Context ablations use causal counts, not whole-sequence division")
        if not isinstance(kda_conv_size, int) or kda_conv_size < 2:
            raise ValueError("kda_conv_size must be an integer >= 2")
        self.kda_conv_size = kda_conv_size
        if output_norm not in ("none", "capped"):
            raise ValueError("output_norm must be none or capped")
        self.output_norm = output_norm
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if sdpa_backend not in ("flash", "auto", "math"):
            raise ValueError("invalid SDPA backend")
        self.variant = variant
        self.linear_threshold = validate_threshold(FIXED_THRESHOLDS.get(variant, linear_threshold))
        self.sdpa_backend = sdpa_backend
        self.require_fused = require_fused
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        self.norm_eps = norm_eps
        kwargs["attention_normalizer"] = "softmax" if variant == "softmax_sdpa" else "relu2max"
        if variant == "kda":
            # These inherited NanoGPT fields are not KDA operations. Persist the
            # actual positional/scaling choices in every HF checkpoint.
            kwargs["use_rotary_embeddings"] = False
            kwargs["use_absolute_position_embeddings"] = False
            kwargs["use_qk_norm_scale"] = False
        super().__init__(**kwargs)
        if self.attention_dropout or self.dropout:
            raise ValueError("This matched experiment uses zero dropout")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("This experiment uses full multi-head attention")
        if variant == "kda" and (self.bias or self.hidden_size // self.num_attention_heads > 256):
            raise ValueError("This KDA arm requires bias=False and head_dim<=256")

    @property
    def effective_threshold(self):
        return FIXED_THRESHOLDS.get(self.variant, self.linear_threshold)

    @property
    def is_context_attention(self):
        return self.variant in CONTEXT_ARMS

"""Opt-in Hugging Face configuration with independent residual and head widths.

This is a separate model type: existing Comparison/NanoGPT checkpoints retain
their architecture and code. Concatenated full multi-head attention is used.
"""
import math
from transformers import PretrainedConfig


class SearchConfig(PretrainedConfig):
    model_type = "nanogpt_flexible_attention_search"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=50304, hidden_size=384, intermediate_size=None,
                 num_hidden_layers=16, num_attention_heads=3, num_key_value_heads=None,
                 n_qk_head_dim=128, n_v_head_dim=128, max_position_embeddings=4096,
                 variant="relu2", sdpa_backend="flash", require_fused=True,
                 relu2max_accelerator="triton", relu2max_divisor=256.0,
                 relu2max_divide_by_sequence_length=False, linear_threshold=16.0,
                 use_qk_norm=True, qk_norm_eps=1e-6, use_qk_norm_scale=True,
                 qk_norm_scale_init=20.0, use_rotary_embeddings=True,
                 rope_length=None, rope_theta=10000.0,
                 use_absolute_position_embeddings=False, norm_eps=1e-6,
                 output_norm="none", hidden_act="gelu", bias=False,
                 dropout=0.0, attention_dropout=0.0, initializer_range=0.02,
                 tie_word_embeddings=True, use_cache=True, **kwargs):
        intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size
        num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        for name, value in dict(vocab_size=vocab_size, hidden_size=hidden_size,
                                intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers,
                                num_attention_heads=num_attention_heads, n_qk_head_dim=n_qk_head_dim,
                                n_v_head_dim=n_v_head_dim, max_position_embeddings=max_position_embeddings).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_key_value_heads != num_attention_heads:
            raise ValueError("Search v1 supports full multi-head attention; GQA is not implemented")
        if max(n_qk_head_dim, n_v_head_dim) > 256:
            raise ValueError("The reused fused kernels support head widths <=256")
        if variant not in ("softmax_sdpa", "relu2", "linear16"):
            raise ValueError("variant must be softmax_sdpa, relu2, or linear16")
        if sdpa_backend not in ("flash", "auto", "math"):
            raise ValueError("sdpa_backend must be flash, auto, or math")
        if relu2max_accelerator not in ("triton", "auto", "torch"):
            raise ValueError("relu2max_accelerator must be triton, auto, or torch")
        if linear_threshold != 16.0:
            raise ValueError("The linear16 arm has an immutable threshold of 16")
        if output_norm not in ("none", "capped"):
            raise ValueError("output_norm must be none or capped")
        if dropout != 0 or attention_dropout != 0:
            raise ValueError("These fused attention comparisons require zero dropout")
        if use_absolute_position_embeddings:
            raise ValueError("Search v1 supports RoPE or no positional encoding, not absolute embeddings")
        # Odd QK widths are legal with an explicitly even partial rotary width.
        rope_length = n_qk_head_dim if rope_length is None and use_rotary_embeddings else (rope_length or 0)
        if not isinstance(rope_length, int) or isinstance(rope_length, bool) or rope_length < 0 or rope_length > n_qk_head_dim or rope_length % 2:
            raise ValueError("rope_length must be even and within n_qk_head_dim")
        for name, value in dict(qk_norm_eps=qk_norm_eps, norm_eps=norm_eps,
                                rope_theta=rope_theta, relu2max_divisor=relu2max_divisor,
                                initializer_range=initializer_range).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(qk_norm_scale_init, (int, float)) or isinstance(qk_norm_scale_init, bool) or not math.isfinite(qk_norm_scale_init):
            raise ValueError("qk_norm_scale_init must be finite")
        fields = locals().copy()
        for key in ("self", "kwargs", "fields", "name", "value", "__class__"):
            fields.pop(key, None)
        for key, value in fields.items():
            setattr(self, key, value)
        self.attention_normalizer = "softmax" if variant == "softmax_sdpa" else "relu2max"
        self.padded_head_dim = max(n_qk_head_dim, n_v_head_dim)
        if kwargs.pop("learned_subtractive_bias", False) or kwargs.pop("causal_length_alpha", 0.0) != 0.0:
            raise ValueError("Search v1 supports the three base attention arms without bias/length-scaling treatments")
        self.learned_subtractive_bias = False
        self.causal_length_alpha = 0.0
        self.causal_length_anchor = kwargs.pop("causal_length_anchor", 512.0)
        # PretrainedConfig owns these public fields and serialization behavior.
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def effective_threshold(self):
        return 16.0

    @property
    def is_context_attention(self):
        return False


SearchConfig.register_for_auto_class()

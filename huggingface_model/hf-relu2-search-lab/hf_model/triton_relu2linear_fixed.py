"""Fully fused attention entry points with immutable transition constants.

The shared tiled forward/backward kernels receive a literal tl.constexpr
threshold, so there is no threshold tensor load or runtime activation branch.
The fixed-8 entry point remains in triton_relu2linear_attention.py.
"""
from .triton_relu2linear_attention import _dispatch


def triton_relu2linear2_attention(q, k, v, scale=None, divisor=256.0,
                                causal=True, causal_offset=None, attention_mask=None):
    """Quadratic until 2; tangent line 4*x - 4 above 2."""
    return _dispatch(q, k, v, scale, divisor, causal, causal_offset, attention_mask, 2.0, False)


def triton_relu2linear4_attention(q, k, v, scale=None, divisor=256.0,
                                causal=True, causal_offset=None, attention_mask=None):
    """Quadratic until 4; tangent line 8*x - 16 above 4."""
    return _dispatch(q, k, v, scale, divisor, causal, causal_offset, attention_mask, 4.0, False)


def triton_relu2linear16_attention(q, k, v, scale=None, divisor=256.0,
                                 causal=True, causal_offset=None, attention_mask=None):
    """Quadratic until 16; tangent line 32*x - 256 above 16."""
    return _dispatch(q, k, v, scale, divisor, causal, causal_offset, attention_mask, 16.0, False)

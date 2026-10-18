"""Exact reference and public experiment definitions for context ablations.

s_ij = scale * dot(q_i, k_j) - bias_head
w_ij = relu(s_ij)^2 / divisor * max(1, visible_keys_i / anchor)^(-alpha)

Bias is an unconstrained signed parameter, initialized to zero. A positive
bias discards weak positive matches. Causal counts respect padding and cache
offsets. No count depends on future padding entries or on query chunk size.
"""
import math
import numbers
import torch

# (learned subtractive bias, length exponent). Names encode the mathematical
# treatment; non-default anchor and bias initialization are recorded in config.
CONTEXT_ARMS = {
    "relu2_bias": (True, 0.0),
    "relu2_scale_half": (False, 0.5),
    "relu2_scale_one": (False, 1.0),
    "relu2_bias_scale_half": (True, 0.5),
    "relu2_bias_scale_one": (True, 1.0),
}


def validate_length_scaling(alpha, anchor):
    if isinstance(alpha, bool) or alpha not in (0.0, 0.5, 1.0):
        raise ValueError("length alpha must be 0, 0.5 or 1")
    if not isinstance(anchor, numbers.Real) or isinstance(anchor, bool) or not math.isfinite(anchor) or anchor <= 0:
        raise ValueError("length anchor must be positive and finite")


def visible_counts(batch, queries, keys, device, attention_mask=None, causal=True, causal_offset=None):
    offset = keys - queries if causal_offset is None else causal_offset
    rows = torch.arange(queries, device=device)
    end = (rows + offset + 1).clamp(0, keys) if causal else torch.full_like(rows, keys)
    if attention_mask is None:
        return end.unsqueeze(0).expand(batch, -1)
    if attention_mask.ndim != 2 or attention_mask.shape[0] != batch or attention_mask.shape[1] < keys:
        raise ValueError("key mask must have shape [batch, >= keys]")
    prefix = torch.nn.functional.pad((attention_mask[:, :keys] != 0).long().cumsum(-1), (1, 0))
    return prefix[:, end]


def length_factors(counts, alpha=0.0, anchor=512.0, dtype=torch.float32):
    validate_length_scaling(alpha, anchor)
    if alpha == 0:
        return torch.ones_like(counts, dtype=dtype)
    ratio = (counts.to(dtype) / anchor).clamp_min(1)
    return ratio.rsqrt() if alpha == 0.5 else ratio.reciprocal()


def context_attention_reference(q, k, v, scale=1.0, bias=None, divisor=256.0,
                                alpha=0.0, anchor=512.0, causal=True,
                                causal_offset=None, attention_mask=None):
    """Autograd reference; FP64 inputs retain FP64 for numerical gradcheck."""
    validate_length_scaling(alpha, anchor)
    if not isinstance(divisor, numbers.Real) or isinstance(divisor, bool) or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be positive and finite")
    b, h, m, _ = q.shape
    n = k.shape[-2]
    offset = n - m if causal_offset is None else causal_offset
    if bias is not None and (bias.shape != (h,) or bias.device != q.device):
        raise ValueError("bias must be a vector of one value per head on Q's device")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        raw = q.to(dtype) @ k.to(dtype).transpose(-1, -2)
        scores = raw * scale
        if bias is not None:
            scores = scores - bias.to(dtype)[None, :, None, None]
        allowed = torch.ones((b, 1, m, n), dtype=torch.bool, device=q.device)
        if causal:
            allowed = allowed & (torch.arange(n, device=q.device)[None, :] <=
                                 torch.arange(m, device=q.device)[:, None] + offset)
        if attention_mask is not None:
            allowed = allowed & (attention_mask[:, None, None, :n] != 0)
        counts = visible_counts(b, m, n, q.device, attention_mask, causal, offset)
        factor = length_factors(counts, alpha, anchor, dtype=dtype)
        # Mask AFTER subtracting bias: a negative bias must not resurrect a
        # masked score, including a row with no allowed keys.
        weights = scores.relu().square().masked_fill(~allowed, 0) / divisor
        weights = weights * factor[:, None, :, None]
        return (weights @ v.to(dtype)).to(q.dtype)

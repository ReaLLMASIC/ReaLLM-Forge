"""Stable tangent-linear ReLU² and validation shared by the new variants."""

import math
import numbers
import struct

import torch


def validate_threshold(threshold):
    """Accept a positive finite host real with representable FP32 coefficients."""
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be a positive finite host real, not a tensor or bool")
    threshold = float(threshold)
    try:
        rounded = struct.unpack("f", struct.pack("f", threshold))[0]
    except (OverflowError, struct.error):
        rounded = float("inf")
    if not math.isfinite(threshold) or rounded <= 0 or not math.isfinite(rounded):
        raise ValueError("threshold must be positive, finite and representable in FP32")
    if rounded * rounded > torch.finfo(torch.float32).max:
        raise ValueError("threshold squared must fit in FP32")
    return threshold


def relu2linear(x, threshold=8.0):
    """0 for x<=0, x² through threshold, then 2*threshold*x-threshold².

    The derivative is 2*min(ReLU(x), threshold), including both join points.
    Unlike a where(x>t, linear, x*x), the square never sees the linear tail.
    FP16/BF16 inputs use FP32 intermediates; the result has the input dtype.
    """
    threshold = validate_threshold(threshold)
    original_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    positive = torch.relu(x)
    capped = positive.clamp(max=threshold)
    return (capped.square() + (2.0 * threshold) * (positive - capped)).to(original_dtype)

"""Utilities for aggregating learnable bit usage across modules."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


def _infer_device(module: nn.Module) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def iter_bit_usage_modules(module: nn.Module) -> Iterable[nn.Module]:
    """Yield submodules that expose a ``bit_usage`` method."""
    for child in module.modules():
        if hasattr(child, "bit_usage") and callable(getattr(child, "bit_usage")):
            yield child


def compute_total_bit_usage(module: nn.Module) -> torch.Tensor:
    """Return the total bit usage reported by all child modules.

    The returned tensor lives on the same device as ``module`` (or CPU if no
    parameters or buffers are present) and participates in autograd, allowing
    gradients to flow back to bit-width parameters.
    """

    device = _infer_device(module)
    total = torch.tensor(0.0, device=device)
    for child in iter_bit_usage_modules(module):
        usage = child.bit_usage()
        if not torch.is_tensor(usage):
            usage = torch.tensor(float(usage), device=device)
        else:
            usage = usage.to(device)
        total = total + usage
    return total


def collect_bit_usage(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a mapping from module names to their reported bit usage."""
    usage: Dict[str, torch.Tensor] = {}
    device = _infer_device(module)
    for name, child in module.named_modules():
        if hasattr(child, "bit_usage") and callable(getattr(child, "bit_usage")):
            value = child.bit_usage()
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=device)
            else:
                value = value.to(device)
            usage[name] = value
    return usage


def count_tracked_parameters(module: nn.Module) -> int:
    """Count parameters covered by bit_usage modules (weight + bias if present)."""
    total = 0
    for child in iter_bit_usage_modules(module):
        w = getattr(child, "weight", None)
        b = getattr(child, "bias", None)
        if w is not None:
            total += w.numel()
        if b is not None:
            total += b.numel()
    return total



__all__ = [
    "collect_bit_usage",
    "compute_total_bit_usage",
    "iter_bit_usage_modules",
    "count_tracked_parameters",
]

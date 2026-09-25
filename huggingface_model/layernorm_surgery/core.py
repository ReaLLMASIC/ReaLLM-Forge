"""Pure tensor utilities shared by the dashboard and sweep runner."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MergeStats:
	threshold: float
	hidden_size: int
	vocab_size: int
	zero_channels: int
	nonzero_channels: int
	shaved_parameters: int
	shaved_fraction: float
	permutation: torch.Tensor
	merged_nonzero: torch.Tensor


def effective_gain(module) -> torch.Tensor:
	"""Return the actual multiplier (Gemma RMSNorm stores multiplier minus one)."""
	weight = module.weight.detach()
	if "gemma" in type(module).__name__.casefold() and "rmsnorm" in type(module).__name__.casefold():
		return weight + 1
	return weight


def set_effective_gain(module, gain: torch.Tensor) -> None:
	"""Set an effective multiplier using a module's native parameterization."""
	stored = gain - 1 if "gemma" in type(module).__name__.casefold() and "rmsnorm" in type(module).__name__.casefold() else gain
	module.weight.copy_(stored)


def discover_norms(model) -> dict[str, torch.nn.Module]:
	"""Find LayerNorm/RMSNorm-like modules with a one-dimensional gain."""
	return {
		name: module
		for name, module in model.named_modules()
		if getattr(module, "weight", None) is not None
		and module.weight.ndim == 1
		and ("norm" in name.casefold() or "norm" in type(module).__name__.casefold())
	}


def threshold_gain(gain: torch.Tensor, threshold: float) -> torch.Tensor:
	"""Zero gains whose magnitude is strictly below ``threshold``."""
	if threshold < 0:
		raise ValueError("threshold must be non-negative")
	return torch.where(gain.abs() < threshold, torch.zeros_like(gain), gain)


def symmetric_quantize(values: torch.Tensor, bits: int) -> torch.Tensor:
	"""Fake-quantize a tensor with one signed symmetric scale."""
	if bits < 2 or bits > 8:
		raise ValueError("bits must be between 2 and 8")
	qmax = (1 << (bits - 1)) - 1
	peak = values.abs().max()
	if peak == 0:
		return values.clone()
	scale = peak / qmax
	return torch.clamp(torch.round(values / scale), -qmax, qmax) * scale


def merge_gain_into_head(
	head: torch.Tensor, gain: torch.Tensor, threshold: float, bits: int | None = None
) -> MergeStats:
	"""Fold a gain into LM-head columns, group zeros, and optionally fake-quantize.

	The returned matrix contains only non-zero columns. ``permutation`` gives the
	new channel order (non-zero first, zero last), so the same permutation can be
	applied to the normalized hidden state before the reduced dot product.
	"""
	if head.ndim != 2 or gain.ndim != 1 or head.shape[1] != gain.numel():
		raise ValueError("head must be [vocab, hidden] and gain must be [hidden]")
	pruned = threshold_gain(gain, threshold)
	nonzero = torch.nonzero(pruned != 0, as_tuple=False).flatten()
	zero = torch.nonzero(pruned == 0, as_tuple=False).flatten()
	permutation = torch.cat((nonzero, zero))
	merged = head[:, nonzero] * pruned[nonzero].unsqueeze(0)
	if bits is not None:
		merged = symmetric_quantize(merged, bits)
	shaved = int(zero.numel() * head.shape[0])
	return MergeStats(
		threshold=float(threshold), hidden_size=int(head.shape[1]), vocab_size=int(head.shape[0]),
		zero_channels=int(zero.numel()), nonzero_channels=int(nonzero.numel()),
		shaved_parameters=shaved, shaved_fraction=shaved / int(head.numel()),
		permutation=permutation, merged_nonzero=merged,
	)

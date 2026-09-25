"""Depth-wise attention residuals.

This module implements Full Attention Residuals: each Transformer sublayer gets
an input selected from the embedding and all earlier sublayer outputs.  Routing
is token-local (the softmax dimension is depth), so sequence mixing remains the
responsibility of the normal self-attention module.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class FullAttentionResidual(nn.Module):
    """Mix earlier sublayer outputs with zero-initialized pseudo-queries."""

    def __init__(
        self,
        n_destinations: int,
        n_embd: int,
        eps: float = 1e-6,
        weight_variant: str = "softmax",
        relu2max_shift: float = 1.0,
    ):
        super().__init__()
        # Includes one destination for each attention/MLP and one for ln_f.
        self.queries = nn.Parameter(torch.zeros(n_destinations, n_embd))
        self.eps = eps
        self.weight_variant = weight_variant
        self.relu2max_shift = relu2max_shift
        if weight_variant not in ("softmax", "relu2max"):
            raise ValueError(f"unknown attention residual weight variant: {weight_variant}")
        if weight_variant == "relu2max" and relu2max_shift <= 0:
            raise ValueError("attention residual relu2max shift must be positive")

    def _weights(self, scores: torch.Tensor) -> torch.Tensor:
        if self.weight_variant == "softmax":
            return scores.softmax(dim=0)

        # Centering retains softmax's invariance to a common score offset.  The
        # positive shift gives zero-initialized queries a uniform distribution
        # with a nonzero gradient; plain relu(scores) ** 2 would be dead at zero.
        centered_scores = scores - scores.mean(dim=0, keepdim=True)
        terms = torch.relu(centered_scores + self.relu2max_shift).square()
        denominator = terms.sum(dim=0, keepdim=True)
        normalized = terms / denominator.clamp_min(torch.finfo(terms.dtype).tiny)

        # Retain a finite fallback for non-finite/extreme inputs and preserve
        # the residual mixer's convex-combination invariant.
        uniform = torch.full_like(terms, 1.0 / terms.size(0))
        return torch.where(denominator > 0, normalized, uniform)

    def forward(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        if not sources:
            raise ValueError("attention residuals require at least one source")
        values = torch.stack(sources, dim=0)  # depth, batch, time, channels
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.queries[destination])
        weights = self._weights(scores)
        return torch.einsum("dbt,dbtc->btc", weights, values)

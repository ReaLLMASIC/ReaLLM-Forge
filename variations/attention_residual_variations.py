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

    def __init__(self, n_destinations: int, n_embd: int, eps: float = 1e-6):
        super().__init__()
        # Includes one destination for each attention/MLP and one for ln_f.
        self.queries = nn.Parameter(torch.zeros(n_destinations, n_embd))
        self.eps = eps

    def forward(self, sources: list[torch.Tensor], destination: int) -> torch.Tensor:
        if not sources:
            raise ValueError("attention residuals require at least one source")
        values = torch.stack(sources, dim=0)  # depth, batch, time, channels
        keys = F.rms_norm(values, (values.size(-1),), eps=self.eps)
        scores = torch.einsum("dbtc,c->dbt", keys, self.queries[destination])
        weights = scores.softmax(dim=0)
        return torch.einsum("dbt,dbtc->btc", weights, values)

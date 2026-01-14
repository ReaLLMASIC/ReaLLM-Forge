"""Learned confidence scaling variations."""

from __future__ import annotations
import torch
import torch.nn as nn


class BaseLearnedConfidence(nn.Module):
    """Applies a learned vector dot product (and optional constant) to scale inputs."""

    def __init__(self, config, prefix: str, init_fn):
        super().__init__()
        # initialize scaling vector
        self.vector = nn.Parameter(init_fn(config.n_embd))
        # optional additive constant
        use_const = getattr(config, f"use_{prefix}_resid_const", False)
        const_val = getattr(config, f"{prefix}_resid_const", 0.0)
        learn_const = getattr(config, f"learn_{prefix}_resid_const", False)
        if use_const:
            const_tensor = torch.tensor(const_val)
            if learn_const:
                self.const = nn.Parameter(const_tensor)
            else:
                self.register_buffer("const", const_tensor)
        else:
            self.const = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (x * self.vector).sum(dim=-1, keepdim=True)
        if self.const is not None:
            scale = scale + self.const
        return x * scale


class ZerosLearnedConfidence(BaseLearnedConfidence):
    def __init__(self, config, prefix: str):
        super().__init__(config, prefix, lambda dim: torch.zeros(dim))


class OnesLearnedConfidence(BaseLearnedConfidence):
    def __init__(self, config, prefix: str):
        super().__init__(config, prefix, lambda dim: torch.ones(dim))


class GaussianLearnedConfidence(BaseLearnedConfidence):
    def __init__(self, config, prefix: str):
        super().__init__(
            config,
            prefix,
            lambda dim: (
                torch.nn.init.normal_(
                    torch.empty(dim),
                    mean=config.resid_gaussian_mean_init,
                    std=config.resid_gaussian_std_init,
                )
            ),
        )


learned_confidence_dictionary = {
    "zeros": ZerosLearnedConfidence,
    "ones": OnesLearnedConfidence,
    "gaussian": GaussianLearnedConfidence,
}

"""Recurrent-block variants for latent-chaining style training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RecurrentBlockConfig:
    """Configuration for recurrent block loss construction."""

    latent_steps: int
    skip_steps: int
    weight_start: float
    weight_end: float


def build_loss_weights(
    config: RecurrentBlockConfig,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Create per-position loss weights with optional skip mask."""
    weights = torch.linspace(
        config.weight_start,
        config.weight_end,
        steps=seq_len,
        device=device,
    ).repeat(batch_size, 1)

    if config.skip_steps:
        weights[:, : config.skip_steps] = 0.0

    return weights


def latent_chaining_block(
    *,
    embed_tokens,
    forward_embedded,
    x_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
    config: RecurrentBlockConfig,
) -> torch.Tensor:
    """
    Compute recurrent-block loss while preserving full self-attention context.

    The hidden buffer grows by one position each step. We either append a
    ground-truth token embedding (teacher-forcing) or reuse the previous
    hidden state (latent chaining), then compute loss on the newest token.
    """
    batch_size, seq_len = x_tokens.shape
    device = x_tokens.device

    weights = build_loss_weights(config, batch_size, seq_len, device)
    nz_sum = weights.sum() + 1e-8

    hidden_buf = None
    hidden_prev = None
    total_loss = 0.0

    for t in range(seq_len):
        use_latent = t >= config.latent_steps and hidden_prev is not None
        if use_latent:
            new_piece = hidden_prev
        else:
            new_piece = embed_tokens(x_tokens[:, t : t + 1])

        hidden_buf = new_piece if hidden_buf is None else torch.cat(
            [hidden_buf, new_piece], dim=1
        )

        logits_all, h_all = forward_embedded(hidden_buf)
        logits_step = logits_all[:, -1, :]
        hidden_prev = h_all[:, -1:, :]

        ce = F.cross_entropy(logits_step, y_tokens[:, t], reduction="none")
        total_loss += (ce * weights[:, t]).sum()

    return total_loss / nz_sum


RECURRENT_BLOCK_VARIANTS = {
    "latent_chaining": latent_chaining_block,
}

"""Utilities for Self Logits Evolution Decoding (SLED).

This module implements the SLED algorithm described in
"SLED: Self Logits Evolution Decoding for Improving Factuality in Large
Language Models" (arXiv:2411.02433).

The implementation follows Algorithm 1 from the paper and operates purely on
logits.  Given the logits from the final layer and a collection of logits from
earlier layers, SLED contrasts these distributions to expose the model's latent
knowledge and nudges the final logits toward it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class SLEDConfig:
    """Configuration container for SLED.

    Attributes
    ----------
    alpha:
        Evolution rate ``α`` controlling the magnitude of the update applied to
        the final logits.
    top_k:
        Number of tokens retained for the self-evolution step.  Tokens outside
        this set are assigned ``eta`` to suppress their probability mass.
    temperature:
        Temperature ``τ`` used to transform logits into probabilities while
        computing the latent distribution.
    eta:
        Logit value applied to tokens outside of the top-``k`` set.  The paper
        recommends using a very negative value so that these tokens are ignored
        during sampling.
    """

    alpha: float = 0.5
    top_k: int = 5
    temperature: float = 1.0
    eta: float = -1e9


def apply_sled(
    final_logits: torch.Tensor,
    early_layer_logits: Sequence[torch.Tensor],
    *,
    alpha: float,
    top_k: int,
    temperature: float,
    eta: float = -1e9,
) -> torch.Tensor:
    """Apply the SLED update to ``final_logits``.

    Parameters
    ----------
    final_logits:
        Tensor of shape ``(B, V)`` holding the final-layer logits for the next
        token.
    early_layer_logits:
        Sequence containing tensors with shape ``(B, V)`` representing logits
        from earlier transformer layers.  Each tensor should correspond to the
        same position as ``final_logits``.
    alpha:
        Evolution rate ``α`` controlling the strength of the update.
    top_k:
        Number of tokens to retain during the evolution step.  If ``top_k`` is
        greater than the vocabulary size, all tokens are retained.
    temperature:
        Temperature ``τ`` used when converting logits into probabilities.
    eta:
        Logit value assigned to tokens outside of the selected ``top_k`` set.

    Returns
    -------
    torch.Tensor
        Updated logits with the same shape as ``final_logits``.
    """

    if not early_layer_logits:
        return final_logits

    if top_k <= 0:
        return final_logits

    # All computations are performed in float32 for numerical stability and
    # converted back to the original dtype at the end.
    original_dtype = final_logits.dtype
    final_logits_f = final_logits.float()
    early_logits_f = [logits.float() for logits in early_layer_logits]

    batch_size, vocab_size = final_logits_f.shape
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("SLED requires a positive temperature")

    k = min(int(top_k), vocab_size)
    topk_values, topk_indices = torch.topk(final_logits_f, k, dim=-1)

    final_probs = F.softmax(final_logits_f / tau, dim=-1)

    layer_scores: list[torch.Tensor] = []
    eps = 1e-12

    for logits_n in early_logits_f:
        diff = logits_n - final_logits_f
        diff_norm = diff.norm(dim=-1).clamp_min(eps)

        probs_n = F.softmax(logits_n / tau, dim=-1)
        diff_dot_probs = (diff * probs_n).sum(dim=-1)
        probs_sq_sum = (probs_n * probs_n).sum(dim=-1)

        diff_i = diff.gather(-1, topk_indices)
        probs_i = probs_n.gather(-1, topk_indices)

        numerator = diff_dot_probs.unsqueeze(-1) - diff_i
        denom_vector = torch.sqrt((probs_sq_sum.unsqueeze(-1) - 2.0 * probs_i + 1.0).clamp_min(eps))
        denom = diff_norm.unsqueeze(-1) * denom_vector

        cos_sim = numerator / denom
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0, posinf=0.0, neginf=0.0)
        cos_sim = torch.clamp(cos_sim, min=0.0)

        layer_scores.append(cos_sim ** 2)

    stacked = torch.stack(layer_scores, dim=0)  # (L, B, K)

    totals = stacked.sum(dim=(0, 2))  # (B,)
    totals = totals.clamp_min(eps)
    latent_topk = stacked.sum(dim=0) / totals.unsqueeze(-1)

    adjusted = final_logits_f.clone()
    if k < vocab_size:
        adjusted.fill_(eta)

    final_probs_topk = final_probs.gather(-1, topk_indices)
    updates = topk_values - (alpha / tau) * (final_probs_topk - latent_topk)
    adjusted.scatter_(-1, topk_indices, updates)

    return adjusted.to(dtype=original_dtype)


__all__ = ["SLEDConfig", "apply_sled"]

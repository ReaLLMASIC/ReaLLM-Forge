"""Manifold-constrained Hyper-Connections (mHC) utilities."""
from __future__ import annotations

import torch
import torch.nn as nn


class ManifoldHyperConnections(nn.Module):
    """Manifold-constrained Hyper-Connections mapping for a single sub-layer."""

    def __init__(self, config):
        super().__init__()
        self.streams = config.mhc_expansion_rate
        self.embed_dim = config.n_embd
        self.rmsnorm_eps = config.mhc_rmsnorm_eps
        self.sinkhorn_iters = config.mhc_sinkhorn_iters

        in_dim = self.streams * self.embed_dim
        out_dim = self.streams
        res_dim = self.streams * self.streams

        self.phi_pre = nn.Linear(in_dim, out_dim, bias=False)
        self.phi_post = nn.Linear(in_dim, out_dim, bias=False)
        self.phi_res = nn.Linear(in_dim, res_dim, bias=False)

        self.bias_pre = nn.Parameter(torch.zeros(out_dim))
        self.bias_post = nn.Parameter(torch.zeros(out_dim))
        self.bias_res = nn.Parameter(torch.zeros(res_dim))

        alpha_init = config.mhc_alpha_init
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x.pow(2), dim=-1, keepdim=True).add(self.rmsnorm_eps).sqrt()
        return x / rms

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
        mat = torch.exp(logits)
        for _ in range(self.sinkhorn_iters):
            mat = mat / (mat.sum(dim=-1, keepdim=True) + 1e-6)
            mat = mat / (mat.sum(dim=-2, keepdim=True) + 1e-6)
        return mat

    def _compute_mappings(self, stream: torch.Tensor):
        b, t, n, c = stream.shape
        stream_flat = stream.reshape(b * t, n * c)
        stream_norm = self._rms_norm(stream_flat)

        pre = self.alpha_pre * self.phi_pre(stream_norm) + self.bias_pre
        post = self.alpha_post * self.phi_post(stream_norm) + self.bias_post
        res = self.alpha_res * self.phi_res(stream_norm) + self.bias_res

        h_pre = torch.sigmoid(pre)
        h_post = 2.0 * torch.sigmoid(post)
        h_res = self._sinkhorn(res.reshape(b * t, n, n))

        dtype = stream.dtype
        return h_pre.to(dtype), h_post.to(dtype), h_res.to(dtype)

    def pre_map(self, stream: torch.Tensor):
        h_pre, h_post, h_res = self._compute_mappings(stream)
        b, t, n, c = stream.shape
        stream_2d = stream.reshape(b * t, n, c)
        pre_weights = h_pre.view(b * t, 1, n)
        x_in = torch.bmm(pre_weights, stream_2d).view(b, t, c)
        cache = (h_post, h_res)
        return x_in, cache

    def post_map(self, stream: torch.Tensor, out: torch.Tensor, cache):
        h_post, h_res = cache
        b, t, n, c = stream.shape
        stream_2d = stream.reshape(b * t, n, c)
        out_2d = out.reshape(b * t, 1, c)

        post_weights = h_post.view(b * t, n, 1)
        post_contrib = torch.bmm(post_weights, out_2d)
        res_contrib = torch.bmm(h_res, stream_2d)

        updated = res_contrib + post_contrib
        return updated.view(b, t, n, c)

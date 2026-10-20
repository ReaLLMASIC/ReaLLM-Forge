"""Tiny causal transformer for per-column categorical numeric forecasting."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .config import LaneConfig
from .geometry import ColumnGeometry, dequantize, quantize


@dataclass
class LaneOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, ...]] = None
    hidden_states: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    auxiliary_mse: Optional[torch.Tensor] = None


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.heads = config.n_heads
        self.head_dim = d // self.heads
        self.dropout = config.dropout
        self.attention_norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        self.mlp_norm = nn.RMSNorm(d)
        self.up = nn.Linear(d, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        qkv = self.qkv(self.attention_norm(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(b, t, d)
        x = x + self.attention_out(attended)
        return x + self.down(F.relu(self.up(self.mlp_norm(x))).square())


class LaneModel(PreTrainedModel):
    config_class = LaneConfig
    base_model_prefix = "lane_model"
    main_input_name = "values"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.columns = nn.ModuleList([ColumnGeometry(c, config.d_model) for c in config.columns])
        self.position = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.RMSNorm(config.d_model)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def quantize(self, values):
        return torch.stack([quantize(values[..., i], c["minimum"], c["maximum"], c["levels"])
                            for i, c in enumerate(self.config.columns)], dim=-1)

    def dequantize(self, indices):
        return torch.stack([dequantize(indices[..., i], c["minimum"], c["maximum"], c["levels"])
                            for i, c in enumerate(self.config.columns)], dim=-1)

    def encode_values(self, values):
        return torch.stack([column.encode(values[..., i]) for i, column in enumerate(self.columns)],
                           dim=0).sum(0) / math.sqrt(len(self.columns))

    def expected_values(self, logits):
        """Differentiable categorical means in each column's physical units."""
        means = []
        for logit, spec in zip(logits, self.config.columns):
            grid = torch.linspace(spec["minimum"], spec["maximum"], spec["levels"],
                                  device=logit.device, dtype=logit.dtype)
            means.append((logit.softmax(-1) * grid).sum(-1))
        return torch.stack(means, dim=-1)

    def forward(self, values, labels=None, target_values=None, return_dict=None, **kwargs):
        """Labels are already aligned next-frame targets; this never shifts them."""
        if values.ndim != 3 or values.shape[-1] != self.config.num_columns:
            raise ValueError("values must have shape [batch, time, num_columns]")
        if values.shape[1] < 1 or values.shape[1] > self.config.context_length:
            raise ValueError("Input time length must be in [1, context_length]")
        if not torch.is_floating_point(values):
            raise ValueError("values must be floating-point numeric observations")
        x = self.encode_values(values)
        x = x + self.position(torch.arange(values.shape[1], device=values.device))[None]
        for block in self.blocks:
            x = block(x)
        hidden = self.final_norm(x)
        logits = tuple(column.logits(hidden) for column in self.columns)
        loss = ce = mse = None
        if labels is not None:
            if labels.shape != values.shape:
                raise ValueError("labels must match values shape and be shifted by the caller")
            losses = []
            for i, logit in enumerate(logits):
                valid = labels[..., i] != -100
                if valid.any():
                    losses.append(F.cross_entropy(logit[valid], labels[..., i][valid]))
            if not losses:
                raise ValueError("At least one target must be valid")
            ce = torch.stack(losses).mean()
            loss = ce
            if self.config.auxiliary_mse_weight:
                target = self.dequantize(labels) if target_values is None else target_values
                terms = []
                for i, (logit, spec) in enumerate(zip(logits, self.config.columns)):
                    grid = torch.linspace(spec["minimum"], spec["maximum"], spec["levels"],
                                          device=logit.device, dtype=logit.dtype)
                    prediction = (logit.softmax(-1) * grid).sum(-1)
                    valid = labels[..., i] != -100
                    if valid.any():
                        # Normalize each physical range to two units before squaring.
                        error = (prediction - target[..., i]) * (2 / (spec["maximum"] - spec["minimum"]))
                        terms.append(error[valid].square().mean())
                mse = torch.stack(terms).mean()
                loss = loss + self.config.auxiliary_mse_weight * mse
        output = LaneOutput(loss=loss, logits=logits, hidden_states=hidden,
                            cross_entropy=ce, auxiliary_mse=mse)
        if return_dict is False:
            return output.to_tuple()
        return output

    @torch.no_grad()
    def forecast(self, context, horizon, mode="mean", temperature=1.0, top_k=0, generator=None):
        """Autoregressive bounded-window forecast; no future truth is consumed.

        Mean feeds back a continuous expected value. Greedy and sampled decoding
        feed back quantized level centers. Recompute the window when it slides;
        this deliberately avoids stale KV caches with learned local positions.
        """
        if mode not in {"mean", "greedy", "sample"}:
            raise ValueError("mode must be mean, greedy, or sample")
        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if temperature <= 0 or top_k < 0:
            raise ValueError("temperature must be positive and top_k nonnegative")
        was_training = self.training
        self.eval()
        history, predictions = context, []
        try:
            for _ in range(horizon):
                output = self(history[:, -self.config.context_length:])
                columns = []
                for logits, spec in zip(output.logits, self.config.columns):
                    scores = logits[:, -1]
                    if mode == "mean":
                        grid = torch.linspace(spec["minimum"], spec["maximum"], spec["levels"],
                                              device=scores.device, dtype=scores.dtype)
                        values = (scores.softmax(-1) * grid).sum(-1)
                    else:
                        if mode == "greedy":
                            ids = scores.argmax(-1)
                        else:
                            scores = scores / temperature
                            if top_k:
                                chosen, indices = scores.topk(min(top_k, scores.shape[-1]), dim=-1)
                                pick = torch.multinomial(chosen.softmax(-1), 1, generator=generator)
                                ids = indices.gather(-1, pick).squeeze(-1)
                            else:
                                ids = torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)
                        values = dequantize(ids, spec["minimum"], spec["maximum"], spec["levels"])
                    columns.append(values)
                next_values = torch.stack(columns, dim=-1)
                predictions.append(next_values)
                history = torch.cat((history[:, -self.config.context_length + 1:] if self.config.context_length > 1 else history[:, :0],
                                     next_values[:, None]), dim=1)
        finally:
            self.train(was_training)
        return torch.stack(predictions, dim=1) if predictions else context[:, :0]

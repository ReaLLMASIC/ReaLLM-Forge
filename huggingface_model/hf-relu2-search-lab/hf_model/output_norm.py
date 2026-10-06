"""Tokenwise radial cap with zero-centered RMSNorm-style channel gains.

For a full decoder branch vector x of width d:
    r = ||x||_2 / sqrt(d)
    y = (1 + gain) * x / max(1, r)
The radial operation is identity for r<=1. Gain is always applied, as the
affine part of RMSNorm; gains can take the final vector outside the ball.
No epsilon is needed: the denominator is bounded below by 1.
"""
import torch
from torch import nn


class CappedHypersphereNorm(nn.Module):
    def __init__(self, width, enabled=True):
        super().__init__()
        self.width = width
        self.enabled = enabled
        self.gain = nn.Parameter(torch.zeros(width)) if enabled else None
        self._record = False
        self._stats = None

    def start_diagnostics(self):
        self._record, self._stats = True, None

    def stop_diagnostics(self):
        self._record = False
        if self._stats is None:
            return None
        stats, self._stats = self._stats, None
        count = stats["count"]
        result = dict(enabled=self.enabled, vectors=count,
            fraction_above_radius=float(stats["above"]) / count,
            mean_input_rms=float(stats["input_sum"]) / count,
            max_input_rms=float(stats["input_max"]),
            mean_output_rms=float(stats["output_sum"]) / count,
            max_output_rms=float(stats["output_max"]),
            output_fraction_above_radius=float(stats["output_above"]) / count)
        if self.gain is not None:
            effective = 1 + self.gain.detach().float()
            result.update(effective_gain_min=float(effective.min()), effective_gain_max=float(effective.max()),
                          effective_gain_mean=float(effective.mean()))
        return result

    def forward(self, x):
        if not self.enabled and not self._record:
            return x
        # FP32 reductions for production dtypes; retain FP64 for mathematical tests.
        work = x if x.dtype == torch.float64 else x.float()
        ms = work.square().mean(dim=-1, keepdim=True)
        if self.enabled:
            # Strict > selects the identity-side subgradient at the kink r=1.
            # Both operands to rsqrt are >=1, including for a zero input vector.
            inv = torch.rsqrt(torch.where(ms > 1, ms, torch.ones_like(ms)))
            y = (work * inv * (1 + self.gain.to(work.dtype))).to(x.dtype)
        else:
            y = x
        if self._record:
            with torch.no_grad():
                r = ms.detach().sqrt()
                yr = y.detach().to(work.dtype).square().mean(dim=-1, keepdim=True).sqrt()
                row = dict(count=r.numel(), above=(ms.detach() > 1).sum(),
                    input_sum=r.sum(), input_max=r.max(), output_sum=yr.sum(),
                    output_max=yr.max(), output_above=(yr > 1).sum())
                if self._stats is None:
                    self._stats = row
                else:
                    for key, value in row.items():
                        self._stats[key] = torch.maximum(self._stats[key], value) if key.endswith("_max") else self._stats[key] + value
        return y


def start_output_diagnostics(model):
    for module in model.modules():
        if isinstance(module, CappedHypersphereNorm):
            module.start_diagnostics()


def stop_output_diagnostics(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, CappedHypersphereNorm):
            stats = module.stop_diagnostics()
            if stats is not None:
                rows.append(dict(module=name, branch="attention" if name.endswith("attn_output_norm") else "ffn", **stats))
    return rows

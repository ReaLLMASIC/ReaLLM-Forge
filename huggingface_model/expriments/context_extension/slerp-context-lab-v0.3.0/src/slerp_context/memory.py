import math
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import unit, slerp, nlerp


class SphericalMemory(nn.Module):
    def __init__(self, heads, dim, cfg):
        super().__init__()
        self.cfg, self.heads, self.dim = cfg, heads, dim
        self.writer = nn.Linear(2 * dim, dim, bias=False)
        self.anchors = nn.Parameter(torch.randn(heads, cfg.slots, dim) / math.sqrt(dim))
        self.routing_log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(dim))))
        self.update_gate = nn.Sequential(nn.Linear(2 * dim + 3, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.update_gate[-1].weight)
        nn.init.zeros_(self.update_gate[-1].bias)
        timescales = torch.tensor([0.5, 0.1, 0.03, 0.01]).repeat((cfg.slots + 3)//4)[:cfg.slots]
        self.slot_bias = nn.Parameter(torch.logit(timescales)[None, :, None].repeat(heads, 1, 1))
        self.read_q = nn.Linear(dim, dim, bias=False)
        self.read_k = nn.Linear(dim, dim, bias=False)
        self.read_v = nn.Linear(dim, dim, bias=False)
        self.read_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, math.log(0.01 / 0.99))
        self.read_log_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(dim))))
        if cfg.method == "fixed":
            self.update_gate.requires_grad_(False)
            self.slot_bias.requires_grad_(False)

    def write(self, k, v, m, lr, valid):
        with torch.autocast(device_type=k.device.type, enabled=False):
            z = self.writer(torch.cat((k.float(), v.float()), -1))
            weights = torch.einsum("hmd,bhcd->bhmc", unit(self.anchors), unit(z))
            weights = (weights * self.routing_log_scale.exp().clamp(max=100)).softmax(-1)
            c = weights @ z
            radius = c.norm(dim=-1, keepdim=True)
            cn = unit(c)
            clr = radius.clamp_min(1e-8).log().clamp(-12, 12)
            dot = (m * cn).sum(-1, keepdim=True).clamp(-1, 1)
            if self.cfg.method == "fixed":
                a = torch.full_like(lr, self.cfg.fixed_alpha)
            else:
                # dot gives both learned methods the same angle information.
                features = torch.cat((m, cn, lr, clr, dot), -1)
                a = torch.sigmoid(self.update_gate(features) + self.slot_bias)
            if self.cfg.method == "ema":
                vec = (1-a) * lr.exp() * m + a * c
                nm = unit(vec)
                nr = vec.norm(dim=-1, keepdim=True).clamp_min(1e-8).log().clamp(-12, 12)
            else:
                update = nlerp if self.cfg.method == "nlerp" else slerp
                nm = update(m, cn, a)
                nr = (1-a)*lr + a*clr
            nonzero = radius > 1e-8
            nm, nr = torch.where(valid, nm, cn), torch.where(valid, nr, clr)
            return (torch.where(nonzero, nm, m), torch.where(nonzero, nr, lr), valid | nonzero)

    def read(self, q, m, lr, valid):
        with torch.autocast(device_type=q.device.type, enabled=False):
            groups = q.shape[1] // self.heads
            if groups > 1:
                m, lr, valid = (t.repeat_interleave(groups, dim=1) for t in (m, lr, valid))
            qn = unit(self.read_q(q.float()))
            kn = unit(self.read_k(m))
            values = self.read_v(lr.exp() * m)
            scores = qn @ kn.transpose(-1, -2) * self.read_log_scale.exp().clamp(max=100)
            mask = valid.squeeze(-1).unsqueeze(-2)
            # all-empty rows return zero without a softmax(-inf,...) NaN.
            weights = scores.masked_fill(~mask, -1e4).softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            return (weights @ values) * torch.sigmoid(self.read_gate(q.float()))


def delta_block(state, keys, values, beta):
    """Exact tokenwise delta updates, evaluated with one triangular solve.

    State is [..., value_dim, key_dim], keys/values are [..., tokens, dim].
    This is an associative comparator, not a reproduction of Gated DeltaNet.
    """
    n = keys.shape[-2]
    gram = keys @ keys.transpose(-1, -2)
    lower = torch.tril(beta * gram, diagonal=-1)
    system = lower + torch.eye(n, device=keys.device, dtype=keys.dtype)
    residual = beta * (values - keys @ state.transpose(-1, -2))
    writes = torch.linalg.solve_triangular(system, residual, upper=False, unitriangular=True)
    return state + writes.transpose(-1, -2) @ keys


class DeltaMemory(nn.Module):
    """Chunk-decayed delta-rule matrix; local attention remains unchanged.

    One decay per evicted chunk, a corrective write for every evicted token.
    The key and value dimensions equal the backbone head dimension. The legacy
    log-radius/valid fields are bookkeeping placeholders for state compatibility.
    """
    def __init__(self, heads, dim, cfg):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.beta = nn.Linear(2 * dim, 1)
        self.decay_logit = nn.Parameter(torch.full((heads, 1, 1), math.log(99.0)))
        self.query = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.read_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, math.log(0.01 / 0.99))
        self.heads = heads

    def write(self, k, v, matrix, unused, valid):
        with torch.autocast(device_type=k.device.type, enabled=False):
            keys = unit(self.key(k.float()))
            values = self.value(v.float())
            beta = torch.sigmoid(self.beta(torch.cat((k.float(), v.float()), -1)))
            prior = matrix * torch.sigmoid(self.decay_logit)
            matrix = delta_block(prior, keys, values, beta)
            return matrix, unused, torch.ones_like(valid)

    def read(self, q, matrix, unused, valid):
        with torch.autocast(device_type=q.device.type, enabled=False):
            groups = q.shape[1] // self.heads
            if groups > 1:
                matrix = matrix.repeat_interleave(groups, dim=1)
            query = unit(self.query(q.float()))
            values = query @ matrix.transpose(-1, -2)
            return self.output(values) * torch.sigmoid(self.read_gate(q.float()))

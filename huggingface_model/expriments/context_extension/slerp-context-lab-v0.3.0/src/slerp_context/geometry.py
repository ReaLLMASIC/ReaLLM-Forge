"""Geometry in FP32 (FP64 preserved for gradcheck). Never sign-flip endpoints."""
import torch
import torch.nn.functional as F


def unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def orthogonal(u):
    # Choose the coordinate least aligned with u; deterministic at exact antipodes.
    e = F.one_hot(u.abs().argmin(-1), u.shape[-1]).to(u.dtype)
    return unit(e - (e * u).sum(-1, keepdim=True) * u)


def slerp(u, v, a):
    u, v = unit(u), unit(v)
    dot = (u * v).sum(-1, keepdim=True).clamp(-1, 1)
    tangent = v - dot * u
    # sqrt clamp avoids undefined gradients in inactive torch.where branches.
    norm = tangent.square().sum(-1, keepdim=True).clamp_min(1e-16).sqrt()
    theta = torch.atan2(norm, dot)
    direction = torch.where(norm > 1e-7, tangent / norm, orthogonal(u))
    curved = torch.cos(a * theta) * u + torch.sin(a * theta) * direction
    linear = unit((1 - a) * u + a * v)
    out = torch.where(dot > 0.9995, linear, curved)
    return unit(out)


def nlerp(u, v, a):
    mixed = (1 - a) * u + a * v
    # Same deterministic antipodal convention for a cancelled chord.
    return torch.where(mixed.square().sum(-1, keepdim=True) > 1e-14,
                       unit(mixed), slerp(u, v, a))


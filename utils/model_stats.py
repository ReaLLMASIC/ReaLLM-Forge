# utils/model_stats.py
import torch
from torch import Tensor
from typing import Dict, Tuple, List, DefaultDict
from collections import defaultdict
import re

def _moments(t: Tensor) -> Dict[str, float]:
    """
    In‑GPU computation of stdev, kurtosis, min, max, abs_max.
    Returns python floats (so we detach only tiny scalars).
    """
    # keep in fp32 for numeric stability
    t_f32 = t.float()
    mean     = torch.mean(t_f32)
    var      = torch.var(t_f32, unbiased=False)
    stdev    = torch.sqrt(var)

    # Fisher kurtosis (zero for N(0,1))
    m4       = torch.mean((t_f32 - mean) ** 4)
    kurtosis = m4 / var**2 - 3.0

    t_min    = torch.min(t_f32)
    t_max    = torch.max(t_f32)
    abs_max  = torch.max(torch.abs(t_f32))

    return dict(
        stdev    = stdev.item(),
        kurtosis = kurtosis.item(),
        max      = t_max.item(),
        min      = t_min.item(),
        abs_max  = abs_max.item(),
    )


def _extract_type(name: str) -> str:
    """Return a simplified tensor type name used for grouping."""
    name = re.sub(r"^_orig_mod\.", "", name)
    name = re.sub(r"^module\.", "", name)
    if name.startswith("transformer."):
        name = name[len("transformer."):]
    name = re.sub(r"h\.[0-9]+\.", "", name)
    name = re.sub(r"\.(weight|bias)$", "", name)
    return name

@torch.no_grad()
def compute_weight_stats(
    model: torch.nn.Module, device: torch.device
) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return per-tensor metrics, overall averages, and per-type kurtosis stats."""
    per_tensor: Dict[str, Dict] = {}
    accum: Dict[str, float] = {k: 0.0 for k in ["stdev", "kurtosis", "max", "min", "abs_max"]}
    n = 0

    type_groups: DefaultDict[str, List[float]] = defaultdict(list)

    for name, p in model.named_parameters():
        if p.requires_grad:
            t = p.detach().to(device)
            s = _moments(t)
            per_tensor[name] = s
            tp = _extract_type(name)
            type_groups[tp].append(s["kurtosis"])
            for k in accum:
                accum[k] += s[k]
            n += 1

    overall = {k: v / n for k, v in accum.items()}

    type_stats = {
        tp: {
            "kurtosis_mean": float(sum(vals) / len(vals)),
            "kurtosis_max": float(max(vals)),
        }
        for tp, vals in type_groups.items()
    }

    return per_tensor, overall, type_stats

@torch.no_grad()
def compute_activation_stats(
    model: torch.nn.Module,
    x: Tensor,
    y: Tensor,
    iter_num: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    One‑off activation scan used inside ``Trainer.estimate_loss``.
    Performs a *single* forward pass with temporary hooks that:
      • run on the requested ``device`` (GPU keeps host RAM flat)
      • compute moments on‑the‑fly and immediately discard the tensor
    Returns per-tensor metrics, overall averages, and statistics grouped by
    tensor type (mean and maximum kurtosis).
    """
    act_stats: Dict[str, Dict] = {}
    overall: Dict[str, float] = {k: 0.0 for k in ["stdev", "kurtosis", "max", "min", "abs_max"]}
    n = 0
    type_groups: DefaultDict[str, List[float]] = defaultdict(list)

    def make_hook(mod_name: str):
        def _hook(_module, _inp, out):
            nonlocal n
            # Work with first tensor output if module returns tuple
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return                          # skip non‑tensor outputs
            s = _moments(t.detach().to(device))
            act_stats[mod_name] = s
            tp = _extract_type(mod_name)
            type_groups[tp].append(s["kurtosis"])
            for k in overall:
                overall[k] += s[k]
            n += 1
            # free ASAP
            del t
        return _hook

    # Register hooks only on *leaf* modules to avoid duplication
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:   # leaf
            handles.append(module.register_forward_hook(make_hook(name)))

    # Forward pass (targets are optional; model may ignore them)
    _ = model(x, y, iter_num=iter_num) if y is not None else model(x)

    # Clean up
    for h in handles:
        h.remove()

    # Guard against empty hook collection
    if n == 0:
        return {}, {k: 0.0 for k in overall}

    overall = {k: v / n for k, v in overall.items()}

    type_stats = {
        tp: {
            "kurtosis_mean": float(sum(vals) / len(vals)),
            "kurtosis_max": float(max(vals)),
        }
        for tp, vals in type_groups.items()
    }

    return act_stats, overall, type_stats

# utils/model_stats.py
import torch
from torch import Tensor
from typing import Dict, Tuple, List, Optional
from rich.table import Table
from rich.console import Console
import math
import csv

def _valid_float(val: Optional[float]) -> bool:
    """Return True if ``val`` is a finite float."""
    return isinstance(val, float) and math.isfinite(val)

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

@torch.no_grad()
def compute_weight_stats(model: torch.nn.Module, device: torch.device) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    per_tensor: Dict[str, Dict] = {}
    keys = ["stdev", "kurtosis", "max", "min", "abs_max"]
    accum  = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}

    for name, p in model.named_parameters():
        if p.requires_grad:                       # skip buffers etc.
            t = p.detach().to(device)             # stays on GPU if asked
            s = _moments(t)
            per_tensor[name] = s
            for k in accum:                       # running mean over tensors
                val = s[k]
                if not _valid_float(val):
                    continue
                accum[k] += val
                counts[k] += 1

    overall = {k: (accum[k] / counts[k]) if counts[k] > 0 else float("nan") for k in accum}
    return per_tensor, overall

@torch.no_grad()
def compute_activation_stats(
    model:      torch.nn.Module,
    x:          Tensor,
    y:          Tensor,
    iter_num:   int,
    device:     torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """
    One‑off activation scan used inside `Trainer.estimate_loss`.
    Performs a *single* forward pass with temporary hooks that:  
      • run on the requested `device` (GPU keeps host RAM flat)  
      • compute moments on‑the‑fly and immediately discard the tensor  
    Returns a dict keyed by module‑path and an overall average.
    """
    act_stats: Dict[str, Dict] = {}
    keys = ["stdev", "kurtosis", "max", "min", "abs_max"]
    sums   = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}

    def make_hook(mod_name: str):
        def _hook(_module, _inp, out):
            # Work with first tensor output if module returns tuple
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return                          # skip non‑tensor outputs
            s = _moments(t.detach().to(device))
            act_stats[mod_name] = s
            for k in sums:
                val = s[k]
                if not _valid_float(val):
                    continue
                sums[k] += val
                counts[k] += 1
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

    # Guard against empty hook collection or no valid stats
    if all(c == 0 for c in counts.values()):
        return act_stats, {k: float("nan") for k in sums}

    overall = {k: (sums[k] / counts[k]) if counts[k] > 0 else float("nan") for k in sums}
    return act_stats, overall


def model_stats_rows(weight_stats: Dict[str, Dict], act_stats: Dict[str, Dict]) -> Tuple[List[str], List[List[float]]]:
    """Return headers and raw rows for the stats table."""
    stat_keys = ["stdev", "kurtosis", "max", "min", "abs_max"]
    headers = ["tensor"] + [f"W {k}" for k in stat_keys] + [f"A {k}" for k in stat_keys]
    rows: List[List[float]] = []
    printed = set()
    for w_name, ws in weight_stats.items():
        module = w_name.rsplit(".", 1)[0]
        as_ = act_stats.get(module)
        row = [w_name]
        for key in stat_keys:
            row.append(ws.get(key, float("nan")))
        for key in stat_keys:
            row.append(as_.get(key, float("nan")) if as_ else float("nan"))
        rows.append(row)
        printed.add(module)

    for mod, as_ in act_stats.items():
        if mod in printed:
            continue
        row = [mod]
        row.extend([float("nan")] * len(stat_keys))
        for key in stat_keys:
            row.append(as_.get(key, float("nan")))
        rows.append(row)

    return headers, rows


def write_model_stats_csv(weight_stats: Dict[str, Dict], act_stats: Dict[str, Dict], path: str) -> None:
    """Save stats table to *path* as CSV."""
    headers, rows = model_stats_rows(weight_stats, act_stats)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def print_model_stats_table(
    weight_stats: Dict[str, Dict],
    act_stats: Dict[str, Dict],
    csv_path: Optional[str] = None,
    console: Console | None = None,
) -> None:
    """Pretty print weight and activation stats side by side using rich.Table.

    If *csv_path* is provided, also save the raw values to that CSV file.
    """
    stat_keys = ["stdev", "kurtosis", "max", "min", "abs_max"]


    def collect_extremes(stats: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
        ext: Dict[str, Tuple[float, float]] = {}
        for key in stat_keys:
            vals: List[float] = []
            for d in stats.values():
                v = d.get(key)
                if not _valid_float(v):
                    continue
                if key in {"stdev", "max", "abs_max"}:
                    vals.append(abs(v))
                else:
                    vals.append(v)
            if not vals:
                ext[key] = (0.0, 0.0)
            else:
                ext[key] = (min(vals), max(vals))
        return ext

    w_extremes = collect_extremes(weight_stats)
    a_extremes = collect_extremes(act_stats)

    # --- helper for colouring values
    def colour(val: Optional[float], key: str, col_ext: Dict[str, Tuple[float, float]]) -> str:
        if not _valid_float(val):
            return "[orange3]nan[/]"

        lo, hi = col_ext[key]

        if hi == lo:
            t = 0.5
        else:
            if key == "min":
                # largest value should be green, smallest most red
                t = (hi - val) / (hi - lo)
            elif key == "kurtosis":
                # negative -> green, positive -> red. Use log scaling for
                # smoother gradation over wide ranges.
                abs_max = max(abs(lo), abs(hi))
                if abs_max == 0:
                    t = 0.5
                else:
                    norm = math.log(1 + abs(val)) / math.log(1 + abs_max)
                    if val >= 0:
                        t = 0.5 + 0.5 * norm
                    else:
                        t = 0.5 - 0.5 * norm
            else:
                base = abs(val)
                t = (base - lo) / (hi - lo)

            t = max(0.0, min(1.0, t))

        r = int(255 * t)
        g = int(255 * (1 - t))
        color = f"#{r:02x}{g:02x}00"
        return f"[{color}]{val:.6f}[/]"

    headers, raw_rows = model_stats_rows(weight_stats, act_stats)

    if csv_path:
        write_model_stats_csv(weight_stats, act_stats, csv_path)

    table = Table(title="Model Statistics", header_style="bold magenta")
    for head in headers:
        table.add_column(head, justify="right" if head != "tensor" else "left", no_wrap=head=="tensor")

    for raw in raw_rows:
        row: List[str] = [str(raw[0])]
        for key_idx, key in enumerate(stat_keys, start=1):
            row.append(colour(raw[key_idx], key, w_extremes))
        offset = 1 + len(stat_keys)
        for key_idx, key in enumerate(stat_keys, start=offset):
            row.append(colour(raw[key_idx], key, a_extremes))

        table.add_row(*row)

    console = console or Console()
    console.print(table)


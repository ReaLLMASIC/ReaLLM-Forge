#!/usr/bin/env python3
"""Lightweight per-vector angular distortion between two checkpoints.

Computes the per-vector cosine similarity and angle (in degrees) between
matching weight tensors in a baseline and a comparison checkpoint.  This is
the O(n) *inter-checkpoint* comparison — it does NOT compute the expensive
O(n^2) intra-tensor pairwise metrics, L2 norm distributions, group
statistics, or histograms that checkpoint_regex_explorer.py provides.

Usable as both a library and a CLI tool::

    # CLI — compare two checkpoints, write CSV
    python checkpoint_angle_comparison.py baseline/ckpt.pt quantized/ckpt.pt \\
        --pattern 'transformer\\.h\\.[0-9]+\\.(attn|mlp).*\\.weight' \\
        --csv angles.csv

    # Library
    from checkpoint_angle_comparison import load_state_dict, compare_angles
    base_sd, emb = load_state_dict("baseline/ckpt.pt")
    quant_sd, _  = load_state_dict("quantized/ckpt.pt")
    stats, records = compare_angles(base_sd, quant_sd, emb, pattern=r".*weight")
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Default regex: attention + MLP weight matrices
DEFAULT_PATTERN = (
    r"transformer\.h\.[0-9]+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))\.weight"
)


# ── Checkpoint loading ──────────────────────────────────────────────────────

def load_state_dict(
    ckpt_path: str, device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    """Load a checkpoint and return (state_dict, embedding_dim).

    Strips the ``_orig_mod.`` compiler prefix when present.  The embedding
    dimension is read from ``model_args["n_embd"]`` if available.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        cleaned[k.removeprefix("_orig_mod.")] = v
    n_embd: Optional[int] = ckpt.get("model_args", {}).get("n_embd")
    return cleaned, n_embd


# ── Core comparison ─────────────────────────────────────────────────────────

def compare_angles(
    baseline_sd: Dict[str, torch.Tensor],
    other_sd: Dict[str, torch.Tensor],
    embedding_dim: int,
    pattern: str = DEFAULT_PATTERN,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Compare per-vector angles between two state dicts.

    For every weight tensor whose name matches *pattern* and that has an
    axis of size *embedding_dim*, the vectors along that axis are compared
    via cosine similarity and the angle (in degrees) is recorded.

    Returns
    -------
    summary : dict
        Aggregate statistics: ``mean_angle``, ``median_angle``,
        ``max_angle``, ``min_angle``, ``mean_cosine``.
        Empty dict when no vectors matched.
    records : list[dict]
        One entry per vector with keys ``parameter``, ``axis``,
        ``vector_index``, ``angle``, ``cosine_similarity``.
    """
    compiled = re.compile(pattern)
    all_angles: List[float] = []
    all_cosines: List[float] = []
    records: List[Dict[str, object]] = []

    for name in baseline_sd:
        if not compiled.search(name):
            continue
        base_t = baseline_sd[name].detach().float()
        other_t = other_sd.get(name)
        if other_t is None or other_t.shape != base_t.shape:
            continue
        other_t = other_t.detach().float()

        for axis, axis_size in enumerate(base_t.shape):
            if axis_size != embedding_dim:
                continue
            base_vecs = base_t.movedim(axis, -1).reshape(-1, embedding_dim)
            other_vecs = other_t.movedim(axis, -1).reshape(-1, embedding_dim)
            if base_vecs.numel() == 0:
                continue

            cos = F.cosine_similarity(base_vecs, other_vecs, dim=-1, eps=1e-8)
            cos = cos.clamp(-1.0, 1.0)
            angles_deg = torch.rad2deg(torch.acos(cos))

            for idx in range(angles_deg.shape[0]):
                a = float(angles_deg[idx].item())
                c = float(cos[idx].item())
                all_angles.append(a)
                all_cosines.append(c)
                records.append({
                    "parameter": name,
                    "axis": axis,
                    "vector_index": idx,
                    "angle": a,
                    "cosine_similarity": c,
                })

    if not all_angles:
        return {}, records

    summary = {
        "mean_angle": statistics.mean(all_angles),
        "median_angle": statistics.median(all_angles),
        "max_angle": max(all_angles),
        "min_angle": min(all_angles),
        "mean_cosine": statistics.mean(all_cosines),
    }
    return summary, records


def write_csv(
    records: List[Dict[str, object]], csv_path: str
) -> str:
    """Write per-vector comparison records to a CSV file."""
    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["parameter", "axis", "vector_index", "angle", "cosine_similarity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "parameter": rec["parameter"],
                "axis": rec["axis"],
                "vector_index": rec["vector_index"],
                "angle": f"{rec['angle']:.8f}",
                "cosine_similarity": f"{rec['cosine_similarity']:.8f}",
            })
    return csv_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight per-vector angle comparison between two checkpoints."
    )
    parser.add_argument("baseline_ckpt", help="Path to the baseline checkpoint file")
    parser.add_argument("compare_ckpt", help="Path to the comparison checkpoint file")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Regex for parameter names to compare (default: attention + MLP weights)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to write per-vector CSV (optional)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for checkpoint loading (default: cpu)",
    )
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline_ckpt}")
    baseline_sd, emb_dim = load_state_dict(args.baseline_ckpt, device=args.device)

    print(f"Loading comparison: {args.compare_ckpt}")
    other_sd, other_emb = load_state_dict(args.compare_ckpt, device=args.device)

    embedding_dim = emb_dim or other_emb
    if embedding_dim is None:
        raise SystemExit("Could not determine embedding dimension from either checkpoint.")

    summary, records = compare_angles(baseline_sd, other_sd, embedding_dim, pattern=args.pattern)

    if not summary:
        print("No matching vectors found.")
        return

    print(f"Compared {len(records)} vectors")
    print(f"  Mean angle:   {summary['mean_angle']:.4f} deg")
    print(f"  Median angle: {summary['median_angle']:.4f} deg")
    print(f"  Min angle:    {summary['min_angle']:.4f} deg")
    print(f"  Max angle:    {summary['max_angle']:.4f} deg")
    print(f"  Mean cosine:  {summary['mean_cosine']:.6f}")

    if args.csv:
        path = write_csv(records, args.csv)
        print(f"Wrote CSV to {path}")


if __name__ == "__main__":
    main()


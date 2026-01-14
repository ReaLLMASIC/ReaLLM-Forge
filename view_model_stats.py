#!/usr/bin/env python3
"""Utility for displaying model statistics CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple, List, Iterable, Optional

from rich.console import Console
from rich.table import Table

from utils.model_stats import print_model_stats_table


def _valid_float(val: Optional[float]) -> bool:
    """Return True if ``val`` is a finite float."""
    return isinstance(val, float) and math.isfinite(val)

STAT_KEYS = ["stdev", "kurtosis", "max", "min", "abs_max"]


def _f(val: str) -> float:
    try:
        fval = float(val)
        return fval if math.isfinite(fval) else float("nan")
    except Exception:
        return float("nan")


def load_csv(path: Path) -> Tuple[List[str], Dict[str, Dict], Dict[str, Dict]]:
    """Return row order, weight stats, and activation stats from *path*."""
    row_order: List[str] = []
    w_stats: Dict[str, Dict] = {}
    a_stats: Dict[str, Dict] = {}
    with path.open() as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            name = row[0]
            row_order.append(name)
            w_vals = [_f(v) for v in row[1 : 1 + len(STAT_KEYS)]]
            a_vals = [_f(v) for v in row[1 + len(STAT_KEYS) :]]

            module = name
            if any(math.isfinite(v) for v in w_vals):
                w_stats[name] = {k: v for k, v in zip(STAT_KEYS, w_vals)}
                module = name.rsplit(".", 1)[0]

            if any(math.isfinite(v) for v in a_vals):
                a_stats[module] = {k: v for k, v in zip(STAT_KEYS, a_vals)}
    return row_order, w_stats, a_stats


def _collect_value_extremes_multi(
    stats_list: Iterable[Dict[str, Dict]], keys: Iterable[str]
) -> Dict[str, Tuple[float, float]]:
    """Return min/max extremes for each *key* pooling over ``stats_list``."""
    ext: Dict[str, List[float]] = {k: [] for k in keys}
    for stats in stats_list:
        for d in stats.values():
            for key in keys:
                v = d.get(key)
                if not _valid_float(v):
                    continue
                if key in {"stdev", "max", "abs_max"}:
                    ext[key].append(abs(v))
                else:
                    ext[key].append(v)

    out: Dict[str, Tuple[float, float]] = {}
    for key, vals in ext.items():
        out[key] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return out


def _colour_delta(val: float, key: str) -> str:
    """Colour a delta value: green for improvement, red for worsening."""
    if not _valid_float(val):
        return "[orange3]nan[/]"

    improving = val < 0 if key != "min" else val > 0
    color = "green" if improving else "red"
    return f"[{color}]{val:.6f}[/]"


def _percent_change(old: float, new: float) -> float:
    if not _valid_float(old) or not _valid_float(new) or old == 0:
        return float("nan")
    return ((new - old) / abs(old)) * 100.0


def _colour_percent(val: float, key: str) -> str:
    if not _valid_float(val):
        return "[orange3]nan[/]"

    improving = val < 0 if key != "min" else val > 0
    color = "green" if improving else "red"
    return f"[{color}]{val:.2f}%[/]"


def _colour_val(val: float, key: str, ext: Dict[str, Tuple[float, float]]) -> str:
    """Colour a raw value using the same logic as training output."""
    if not _valid_float(val):
        return "[orange3]nan[/]"

    lo, hi = ext[key]
    if hi == lo:
        t = 0.5
    else:
        if key == "min":
            t = (hi - val) / (hi - lo)
        elif key == "kurtosis":
            abs_max = max(abs(lo), abs(hi))
            if abs_max == 0:
                t = 0.5
            else:
                norm = math.log(1 + abs(val)) / math.log(1 + abs_max)
                t = 0.5 + 0.5 * norm if val >= 0 else 0.5 - 0.5 * norm
        else:
            base = abs(val)
            t = (base - lo) / (hi - lo)

        t = max(0.0, min(1.0, t))

    r = int(255 * t)
    g = int(255 * (1 - t))
    color = f"#{r:02x}{g:02x}00"
    return f"[{color}]{val:.6f}[/]"


def print_delta(
    row_order: List[str],
    w1: Dict[str, Dict],
    a1: Dict[str, Dict],
    w2: Dict[str, Dict],
    a2: Dict[str, Dict],
    stats: Iterable[str],
) -> None:
    """Display differences between two model stats tables."""

    # compute diffs
    diff_w: Dict[str, Dict] = {}
    diff_a: Dict[str, Dict] = {}

    for name in set(w1) | set(w2):
        diff_w[name] = {}
        for k in stats:
            v1 = w1.get(name, {}).get(k, float("nan"))
            v2 = w2.get(name, {}).get(k, float("nan"))
            diff_w[name][k] = v2 - v1

    for name in set(a1) | set(a2):
        diff_a[name] = {}
        for k in stats:
            v1 = a1.get(name, {}).get(k, float("nan"))
            v2 = a2.get(name, {}).get(k, float("nan"))
            diff_a[name][k] = v2 - v1

    w_ext = _collect_value_extremes_multi([w1, w2], stats)
    a_ext = _collect_value_extremes_multi([a1, a2], stats)

    console = Console()
    table = Table(title="Δ Model Statistics", header_style="bold magenta")
    table.add_column("Tensor", no_wrap=True)
    for key in stats:
        table.add_column(f"W {key} 1", justify="right")
        table.add_column(f"W {key} 2", justify="right")
        table.add_column(f"ΔW {key}", justify="right")
        table.add_column(f"ΔW {key}%", justify="right")
    for key in stats:
        table.add_column(f"A {key} 1", justify="right")
        table.add_column(f"A {key} 2", justify="right")
        table.add_column(f"ΔA {key}", justify="right")
        table.add_column(f"ΔA {key}%", justify="right")

    # Ensure deterministic row ordering
    all_names = list(row_order)
    for n in set(w2) | set(w1):
        if n not in row_order:
            all_names.append(n)
    printed = set()

    for name in all_names:
        module = name.rsplit(".", 1)[0]
        row = [name]
        for k in stats:
            w_v1 = w1.get(name, {}).get(k, float("nan"))
            w_v2 = w2.get(name, {}).get(k, float("nan"))
            delta_w = diff_w.get(name, {}).get(k, float("nan"))
            pct_w = _percent_change(w_v1, w_v2)
            row.append(_colour_val(w_v1, k, w_ext))
            row.append(_colour_val(w_v2, k, w_ext))
            row.append(_colour_delta(delta_w, k))
            row.append(_colour_percent(pct_w, k))

        for k in stats:
            a_v1 = a1.get(module, {}).get(k, float("nan"))
            a_v2 = a2.get(module, {}).get(k, float("nan"))
            delta_a = diff_a.get(module, {}).get(k, float("nan"))
            pct_a = _percent_change(a_v1, a_v2)
            row.append(_colour_val(a_v1, k, a_ext))
            row.append(_colour_val(a_v2, k, a_ext))
            row.append(_colour_delta(delta_a, k))
            row.append(_colour_percent(pct_a, k))

        table.add_row(*row)
        printed.add(module)

    # Add activation-only modules not seen above
    for mod in diff_a.keys():
        if mod in printed:
            continue
        row = [mod]
        for k in stats:
            row.extend([
                "nan",
                "nan",
                _colour_delta(float("nan"), k),
                _colour_percent(float("nan"), k),
            ])
        for k in stats:
            a_v1 = a1.get(mod, {}).get(k, float("nan"))
            a_v2 = a2.get(mod, {}).get(k, float("nan"))
            delta_a = diff_a.get(mod, {}).get(k, float("nan"))
            pct_a = _percent_change(a_v1, a_v2)
            row.append(_colour_val(a_v1, k, a_ext))
            row.append(_colour_val(a_v2, k, a_ext))
            row.append(_colour_delta(delta_a, k))
            row.append(_colour_percent(pct_a, k))
        table.add_row(*row)

    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="View model stats CSV")
    ap.add_argument("csv", nargs="+", help="One or two CSV files")
    ap.add_argument(
        "--stats",
        default="all",
        help="Comma-separated list of stats to compare (default: all)",
    )
    args = ap.parse_args()

    selected = (
        STAT_KEYS
        if args.stats.lower() == "all"
        else [s for s in args.stats.split(",") if s in STAT_KEYS]
    )

    if len(args.csv) == 1:
        _, w, a = load_csv(Path(args.csv[0]))
        print_model_stats_table(w, a)
    else:
        rows1, w1, a1 = load_csv(Path(args.csv[0]))
        rows2, w2, a2 = load_csv(Path(args.csv[1]))
        row_order = rows1
        for r in rows2:
            if r not in row_order:
                row_order.append(r)
        print_delta(row_order, w1, a1, w2, a2, selected)


if __name__ == "__main__":
    main()


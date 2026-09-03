#!/usr/bin/env python3
"""
plot_multi_script_languages.py

Plot ONLY languages that appear in multiple scripts, grouped by language,
with one bar per (language, script) variant, and bars colored by script.

Fixed color mapping requested:
  - Latn => blue
  - Arab => green
  - Deva => orange
All other scripts get deterministic fallback colors from matplotlib's cycle.

Input JSON format: list of dicts containing at least:
  {"size":"383K", "name":"text_kas_Arab.txt", ...}

Usage:
  python3 plot_multi_script_languages.py --json files.json
  python3 plot_multi_script_languages.py --json files.json --out multi_script.png
  python3 plot_multi_script_languages.py --min-scripts 2 --sort total_kb
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Expected filename: text_<lang>_<script>.txt
FNAME_RE = re.compile(r"^text_([a-z]{3})_([A-Za-z]{4})\.txt$")


def parse_size_to_kb(size_str: str) -> float:
    """
    Convert ls -h-ish sizes to KB (float):
      "383K" -> 383
      "1.2M" -> 1228.8
      "900"  -> treated as bytes -> 0.8789 KB (rare)
    """
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)(B?)\s*$", size_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized size string: {size_str!r}")

    val = float(m.group(1))
    unit = m.group(2).upper()

    mult = {
        "": 1.0 / 1024.0,  # bytes -> KB
        "K": 1.0,
        "M": 1024.0,
        "G": 1024.0**2,
        "T": 1024.0**3,
        "P": 1024.0**4,
    }[unit]
    return val * mult


def script_to_fixed_color(script: str) -> str | None:
    """
    Fixed, semantically meaningful colors.
    Using Matplotlib's classic hexes for consistent look.
    """
    fixed = {
        "Latn": "#1f77b4",  # blue
        "Arab": "#2ca02c",  # green
        "Deva": "#ff7f0e",  # orange
    }
    return fixed.get(script)


def get_fallback_palette() -> List[str]:
    palette = plt.rcParams.get("axes.prop_cycle", None)
    if palette is not None:
        colors = palette.by_key().get("color", [])
        if colors:
            return list(colors)
    # last-resort fallback
    return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def build_script_color_map(scripts_in_use: List[str]) -> Dict[str, str]:
    """
    Deterministic mapping:
      - apply fixed colors first
      - remaining scripts assigned in sorted order from fallback palette
    """
    palette = get_fallback_palette()
    out: Dict[str, str] = {}

    # fixed first
    remaining = []
    for s in scripts_in_use:
        fx = script_to_fixed_color(s)
        if fx is not None:
            out[s] = fx
        else:
            remaining.append(s)

    # deterministic assignment for remaining scripts
    remaining_sorted = sorted(set(remaining))
    for i, s in enumerate(remaining_sorted):
        out[s] = palette[i % len(palette)]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="files.json", help="Input JSON listing")
    ap.add_argument("--out", default=None, help="If set, save figure here (png/pdf/etc)")
    ap.add_argument(
        "--min-scripts",
        type=int,
        default=2,
        help="Keep languages with >= this many distinct scripts (default 2)",
    )
    ap.add_argument(
        "--sort",
        choices=["total_kb", "lang"],
        default="total_kb",
        help="Order language blocks by total size or alphabetically",
    )
    ap.add_argument(
        "--top-langs",
        type=int,
        default=0,
        help="If >0, keep only top N multi-script languages by total KB",
    )
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Aggregate KB by (lang, script)
    lang_script_kb: Counter[Tuple[str, str]] = Counter()
    lang_to_scripts: Dict[str, set] = defaultdict(set)

    for r in rows:
        name = r.get("name", "")
        m = FNAME_RE.match(name)
        if not m:
            continue
        lang, script = m.groups()
        kb = parse_size_to_kb(str(r["size"]))
        lang_script_kb[(lang, script)] += kb
        lang_to_scripts[lang].add(script)

    # Identify languages that appear in multiple scripts
    multi_langs = [lang for lang, scripts in lang_to_scripts.items() if len(scripts) >= args.min_scripts]
    if not multi_langs:
        raise SystemExit("No multi-script languages found in this files.json (given --min-scripts).")

    # Total KB per language (for sorting)
    lang_total_kb: Dict[str, float] = {}
    for lang in multi_langs:
        lang_total_kb[lang] = sum(lang_script_kb[(lang, s)] for s in lang_to_scripts[lang])

    # Sort language blocks
    if args.sort == "total_kb":
        multi_langs.sort(key=lambda l: lang_total_kb[l], reverse=True)
    else:
        multi_langs.sort()

    # Optional: top N languages
    if args.top_langs and args.top_langs > 0:
        multi_langs = multi_langs[: args.top_langs]

    # Scripts used among these multi-script languages
    scripts_in_use: List[str] = []
    for lang in multi_langs:
        for s in lang_to_scripts[lang]:
            if s not in scripts_in_use:
                scripts_in_use.append(s)

    script_color = build_script_color_map(scripts_in_use)

    # Build plotting rows: one bar per (lang, script), grouped by lang
    y_labels: List[str] = []
    x_vals: List[float] = []
    colors: List[str] = []

    # For labeling language blocks and separator lines
    lang_midpoints: List[Tuple[str, float]] = []
    separators: List[int] = []
    y_pos = 0

    for lang in multi_langs:
        scripts = sorted(lang_to_scripts[lang], key=lambda s: lang_script_kb[(lang, s)], reverse=True)
        start = y_pos

        for s in scripts:
            y_labels.append(f"{lang}_{s}")
            x_vals.append(lang_script_kb[(lang, s)])
            colors.append(script_color.get(s, "C0"))
            y_pos += 1

        end = y_pos
        lang_midpoints.append((lang, (start + end - 1) / 2))
        separators.append(end)

    # Plot
    fig_h = max(5, 0.35 * len(y_labels) + 1.5)
    plt.figure(figsize=(12, fig_h))
    plt.barh(range(len(y_labels)), x_vals, color=colors)
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel("Total size (KB)")
    plt.title("Multi-script languages: grouped by language, colored by script")

    # Separators between language blocks
    for cut in separators[:-1]:
        plt.axhline(cut - 0.5, linewidth=1)

    # Language labels on left, centered per block
    for lang, mid in lang_midpoints:
        plt.text(
            0,
            mid,
            f"  {lang}",
            va="center",
            ha="left",
            fontsize=10,
            transform=plt.gca().get_yaxis_transform(),  # x in axes coords, y in data coords
        )

    # Legend (scripts), deterministic order: fixed first, then alphabetical
    fixed_order = ["Latn", "Arab", "Deva"]
    rest = sorted([s for s in script_color.keys() if s not in fixed_order])
    legend_scripts = [s for s in fixed_order if s in script_color] + rest

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=script_color[s], label=s)
        for s in legend_scripts
    ]
    plt.legend(handles=handles, title="Script", loc="lower right")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()


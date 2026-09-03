#!/usr/bin/env python3
"""
plot_tokenization_vs_original.py

Reads the *annotated* filtered JSON (from tokenize_and_annotate_sizes.py),
and plots either:

  (A) tokenized size in KB (default: tiktoken), OR
  (B) ratio of tokenized/original (default), OR
  (C) ratio of original/tokenized (optional)

in the SAME grouped order + SAME color semantics as your previous plots.

Grouping options:
  --group-by {region, script, family}
Coloring options:
  --color-by {region, script, family} (default: same as group-by)

If you color by script:
  - Latn = blue
  - Arab = green
  - Deva = orange
  - others fall back deterministically to matplotlib cycle

Input entry format expected (per row):
{
  "language": "ace",
  "script": "Latn",
  "lang_script": "ace_Latn",
  "size_kb": 277.0,
  "tokenized_sizes": {"tiktoken": 300.0, "byte": 277.0},
  "filename": "text_ace_Latn.txt"
}

Examples:
  # plot tokenized KB (tiktoken) grouped by region
  python3 plot_tokenization_vs_original.py --json filtered_scripts.json --mode tokenized_kb --method tiktoken

  # plot ratio (tiktoken/original) grouped by family, colored by family
  python3 plot_tokenization_vs_original.py --json filtered_scripts.json --mode ratio --method tiktoken --group-by family --color-by family

  # plot ratio (original/tiktoken) grouped by script, colored by script
  python3 plot_tokenization_vs_original.py --mode ratio --ratio-kind orig_over_tok --group-by script --color-by script
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def script_to_region(script: str) -> str:
    mena = {"Arab", "Hebr"}
    south_asia = {"Deva", "Beng", "Gujr", "Guru", "Orya", "Knda", "Mlym"}
    east_asia = {"Jpan", "Hang", "Hani", "Hans", "Hant"}
    se_asia = {"Khmr", "Laoo", "Mymr", "Thai"}
    eurasia = {"Cyrl", "Grek", "Armn", "Geor"}
    horn_africa = {"Ethi"}
    himalaya = {"Tibt"}

    if script in mena:
        return "MENA"
    if script in south_asia:
        return "South Asia"
    if script in east_asia:
        return "East Asia"
    if script in se_asia:
        return "Southeast Asia"
    if script in eurasia:
        return "Eurasia"
    if script in horn_africa:
        return "Horn of Africa"
    if script in himalaya:
        return "Himalaya"
    if script == "Latn":
        return "Latin (global)"
    return f"Other ({script})"


def script_to_family(script: str) -> str:
    # Coarser “language grouping” you asked for earlier
    semitic_scripts = {"Arab", "Hebr", "Ethi"}  # per your earlier preference
    han_scripts = {"Hans", "Hant", "Hani"}
    japanese = {"Jpan"}
    korean = {"Hang"}
    indic = {"Deva", "Beng", "Gujr", "Guru", "Orya", "Knda", "Mlym", "Taml", "Telu", "Sinh"}
    se_asia = {"Khmr", "Laoo", "Mymr", "Thai"}
    cyrillic = {"Cyrl"}
    greek = {"Grek"}
    caucasus = {"Armn", "Geor"}
    tibetan = {"Tibt"}

    if script in semitic_scripts:
        return "Semitic scripts (Arab/Hebr/Ethi)"
    if script in han_scripts:
        return "Han scripts (Hans/Hant/Hani)"
    if script in japanese:
        return "Japanese (Jpan)"
    if script in korean:
        return "Korean (Hang)"
    if script in indic:
        return "Indic scripts"
    if script in se_asia:
        return "Mainland SEA scripts"
    if script in cyrillic:
        return "Cyrillic"
    if script in greek:
        return "Greek"
    if script in caucasus:
        return "Caucasus scripts (Armn/Geor)"
    if script in tibetan:
        return "Tibetan (Tibt)"
    if script == "Latn":
        return "Latin"
    return f"Other ({script})"


def script_to_fixed_color(script: str) -> str | None:
    # fixed colors you wanted
    fixed = {
        "Latn": "#1f77b4",  # blue
        "Arab": "#2ca02c",  # green
        "Deva": "#ff7f0e",  # orange
    }
    return fixed.get(script)


def get_palette() -> List[str]:
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        arr = colors.by_key().get("color", [])
        if arr:
            return list(arr)
    return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def build_color_map(labels_in_order: List[str], mode: str, script_for_key: Dict[str, str]) -> Dict[str, str]:
    """
    Build a deterministic color map for color-by labels.

    If color mode is 'script', apply fixed mapping for Latn/Arab/Deva.
    Otherwise use palette in first-seen order.
    """
    palette = get_palette()
    out: Dict[str, str] = {}

    if mode == "script":
        # fixed first for known scripts
        # labels_in_order here are script names (e.g. Latn, Arab)
        fallback_i = 0
        for lab in labels_in_order:
            fx = script_to_fixed_color(lab)
            if fx is not None:
                out[lab] = fx
            else:
                out[lab] = palette[fallback_i % len(palette)]
                fallback_i += 1
        return out

    # region/family: first-seen order
    for i, lab in enumerate(labels_in_order):
        out[lab] = palette[i % len(palette)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="filtered_scripts.json", help="Annotated filtered json")
    ap.add_argument("--out", default=None, help="Save plot to this path (png/pdf/etc)")

    ap.add_argument("--method", default="tiktoken",
                    help="Which tokenization in tokenized_sizes to use (default: tiktoken)")
    ap.add_argument("--mode", choices=["tokenized_kb", "ratio"], default="ratio",
                    help="Plot tokenized KB or ratio relative to original")
    ap.add_argument("--ratio-kind", choices=["tok_over_orig", "orig_over_tok"], default="tok_over_orig",
                    help="If --mode ratio: which ratio to plot")

    ap.add_argument("--group-by", choices=["region", "script", "family"], default="region",
                    help="How to group entries on the Y axis (blocks)")
    ap.add_argument("--color-by", choices=["region", "script", "family"], default=None,
                    help="How to color bars (default: same as group-by)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="If >0, plot only top N rows by plotted value within the overall list")

    ap.add_argument("--skip-missing", action="store_true",
                    help="Skip rows missing tokenized_sizes[method] (default: skip anyway, but quieter)")
    ap.add_argument("--epsilon", type=float, default=1e-9,
                    help="Small value to avoid divide-by-zero if size_kb is 0")

    args = ap.parse_args()
    if args.color_by is None:
        args.color_by = args.group_by

    with open(args.json, "r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    # Build per-row derived fields
    items: List[Tuple[str, float, str, str, str]] = []
    # tuple: (lang_script, value, script, region, family)
    missing = 0

    for r in rows:
        lang_script = r.get("lang_script")
        script = r.get("script")
        if not lang_script or not script:
            # allow reconstruction if needed
            lang = r.get("language")
            if lang and script:
                lang_script = f"{lang}_{script}"
            else:
                continue

        original_kb = float(r.get("size_kb", 0.0))
        tok_map = r.get("tokenized_sizes", {})
        tok_kb = None
        if isinstance(tok_map, dict):
            v = tok_map.get(args.method)
            if v is not None:
                tok_kb = float(v)

        if tok_kb is None:
            missing += 1
            if not args.skip_missing:
                pass
            continue

        if args.mode == "tokenized_kb":
            value = tok_kb
        else:
            denom = original_kb if args.ratio_kind == "tok_over_orig" else tok_kb
            num = tok_kb if args.ratio_kind == "tok_over_orig" else original_kb
            value = num / max(denom, args.epsilon)

        region = script_to_region(script)
        family = script_to_family(script)
        items.append((lang_script, value, script, region, family))

    if not items:
        raise SystemExit(f"No rows had tokenized_sizes['{args.method}']. Missing={missing}")

    # Group label helpers
    def pick_label(which: str, script: str, region: str, family: str) -> str:
        if which == "script":
            return script
        if which == "region":
            return region
        if which == "family":
            return family
        raise ValueError(which)

    # Group into blocks, order blocks by total value, and within block by value desc
    groups = defaultdict(list)  # label -> list[(lang_script, value, script, region, family)]
    for (ls, val, sc, reg, fam) in items:
        g = pick_label(args.group_by, sc, reg, fam)
        groups[g].append((ls, val, sc, reg, fam))

    group_order = sorted(groups.keys(), key=lambda g: sum(x[1] for x in groups[g]), reverse=True)

    ordered: List[Tuple[str, float, str, str, str]] = []
    for g in group_order:
        ordered.extend(sorted(groups[g], key=lambda x: x[1], reverse=True))

    # Optional top-n by plotted value (after global ordering)
    if args.top_n and args.top_n > 0:
        ordered = ordered[: args.top_n]

    # Build color label list in first-seen order (for deterministic legend mapping)
    color_labels_seen: List[str] = []
    script_for_key: Dict[str, str] = {}
    ordered_color_labels: List[str] = []

    for (ls, val, sc, reg, fam) in ordered:
        cl = pick_label(args.color_by, sc, reg, fam)
        ordered_color_labels.append(cl)
        if cl not in color_labels_seen:
            color_labels_seen.append(cl)
        script_for_key[ls] = sc

    color_map = build_color_map(color_labels_seen, args.color_by, script_for_key)
    bar_colors = [color_map[cl] for cl in ordered_color_labels]

    # Plot
    labels = [x[0] for x in ordered]
    values = [x[1] for x in ordered]
    scripts = [x[2] for x in ordered]
    regions = [x[3] for x in ordered]
    families = [x[4] for x in ordered]

    fig_h = max(6, 0.28 * len(labels) + 1.5)
    plt.figure(figsize=(12, fig_h))

    y = range(len(labels))
    plt.barh(y, values, color=bar_colors)
    plt.yticks(y, labels)

    if args.mode == "tokenized_kb":
        plt.xlabel(f"Tokenized size (KB): {args.method}")
        title = f"Tokenized size (KB) | method={args.method} | grouped by {args.group_by} | colored by {args.color_by}"
    else:
        if args.ratio_kind == "tok_over_orig":
            plt.xlabel(f"Ratio: {args.method} / original (KB)")
        else:
            plt.xlabel(f"Ratio: original / {args.method} (KB)")
        title = f"Tokenization ratio | method={args.method} | grouped by {args.group_by} | colored by {args.color_by}"

    plt.title(title)

    # Add group separators + group labels (same style as before)
    # Recompute block boundaries based on the currently plotted rows
    def row_group_label(i: int) -> str:
        sc, reg, fam = scripts[i], regions[i], families[i]
        return pick_label(args.group_by, sc, reg, fam)

    starts: List[Tuple[str, int, int]] = []  # (g, start, end)
    if labels:
        cur_g = row_group_label(0)
        start = 0
        for i in range(1, len(labels)):
            g = row_group_label(i)
            if g != cur_g:
                starts.append((cur_g, start, i))
                cur_g = g
                start = i
        starts.append((cur_g, start, len(labels)))

    for (g, s, e) in starts:
        if s > 0:
            plt.axhline(s - 0.5, linewidth=1)
        mid = (s + e - 1) / 2
        plt.text(
            0,
            mid,
            f"  {g}",
            va="center",
            ha="left",
            fontsize=10,
            transform=plt.gca().get_yaxis_transform(),
        )

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color_map[lab], label=lab)
        for lab in color_labels_seen
    ]
    plt.legend(handles=handles, title=args.color_by, loc="lower right")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()


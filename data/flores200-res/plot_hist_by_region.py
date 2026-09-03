#!/usr/bin/env python3
import json
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# filename pattern: text_<lang>_<script>.txt
FNAME_RE = re.compile(r"^text_([a-z]{3})_([A-Za-z]{4})\.txt$")

def parse_size_to_kb(size_str: str) -> float:
    """
    Convert "383K", "1.2M" etc to KB (float). Assumes ls -h style.
    """
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)(B?)\s*$", size_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized size: {size_str!r}")
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "": 1.0 / 1024.0,  # bytes -> KB
        "K": 1.0,
        "M": 1024.0,
        "G": 1024.0 * 1024.0,
        "T": 1024.0 * 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0 * 1024.0,
    }[unit]
    return val * mult

def script_to_region(script: str) -> str:
    """
    Region bucket inferred mainly from writing system.
    (Swap this for true geo regions if you prefer.)
    """
    mena = {"Arab", "Hebr"}
    south_asia = {"Deva", "Beng", "Gujr", "Guru", "Orya", "Knda", "Mlym", "Taml", "Telu", "Sinh"}
    east_asia = {"Jpan", "Hang", "Hani"}
    se_asia = {"Khmr", "Laoo", "Mymr", "Thai"}
    eurasia = {"Cyrl", "Grek", "Armn", "Geor"}
    horn_africa = {"Ethi"}
    himalaya = {"Tibt"}

    script = script.strip()
    if script in mena:
        return "MENA (Arab/Hebr scripts)"
    if script in south_asia:
        return "South Asia (Indic scripts)"
    if script in east_asia:
        return "East Asia (CJK scripts)"
    if script in se_asia:
        return "Southeast Asia (SEA scripts)"
    if script in eurasia:
        return "Eurasia (Cyrl/Grek/Armn/Geor)"
    if script in horn_africa:
        return "Horn of Africa (Ethi)"
    if script in himalaya:
        return "Himalaya (Tibt)"
    if script == "Latn":
        return "Latn (global)"
    return f"Other ({script})"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="files.json", help="Input JSON (your listing)")
    ap.add_argument("--out", default=None, help="If set, save figure to this path (png/pdf/etc)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="If >0, keep only top N languages by KB (after aggregation)")
    ap.add_argument("--mixed-threshold", type=float, default=0.25,
                    help="If a language has >=2 regions each with >= this fraction of its total KB, label as Mixed")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Aggregate total KB per language, and also KB per (language, region)
    lang_total_kb = Counter()
    lang_region_kb = defaultdict(Counter)

    for r in rows:
        name = r.get("name", "")
        m = FNAME_RE.match(name)
        if not m:
            continue
        lang, script = m.group(1), m.group(2)
        region = script_to_region(script)
        kb = parse_size_to_kb(r["size"])

        lang_total_kb[lang] += kb
        lang_region_kb[lang][region] += kb

    if not lang_total_kb:
        raise SystemExit("No matching files found. Expected names like text_<lang>_<script>.txt")

    # Decide one region label per language:
    # - normally: dominant region by KB
    # - if "mixed": when two+ regions each exceed threshold fraction
    lang_region_label = {}
    for lang, reg_kb in lang_region_kb.items():
        total = sum(reg_kb.values())
        # sort regions by size
        regs_sorted = sorted(reg_kb.items(), key=lambda kv: kv[1], reverse=True)
        # mixed check
        big_regs = [reg for reg, kb in regs_sorted if (kb / total) >= args.mixed_threshold]
        if len(big_regs) >= 2:
            lang_region_label[lang] = "Mixed"
        else:
            lang_region_label[lang] = regs_sorted[0][0]  # dominant region

    # optionally keep only top-N languages overall
    langs = list(lang_total_kb.keys())
    if args.top_n and args.top_n > 0:
        langs = [l for l, _ in lang_total_kb.most_common(args.top_n)]

    # group languages by region label, then sort regions by total KB descending
    region_to_langs = defaultdict(list)
    for lang in langs:
        region_to_langs[lang_region_label[lang]].append(lang)

    region_order = sorted(
        region_to_langs.keys(),
        key=lambda reg: sum(lang_total_kb[l] for l in region_to_langs[reg]),
        reverse=True
    )

    # within each region, sort langs by KB descending
    ordered_langs = []
    ordered_regions = []
    for reg in region_order:
        ls = sorted(region_to_langs[reg], key=lambda l: lang_total_kb[l], reverse=True)
        ordered_langs.extend(ls)
        ordered_regions.extend([reg] * len(ls))

    # assign each region label a unique color via matplotlib default cycle
    unique_regions = []
    for rlab in ordered_regions:
        if rlab not in unique_regions:
            unique_regions.append(rlab)

    # color mapping (no custom colors specified; uses default cycle)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    region_color = {reg: color_cycle[i % len(color_cycle)] for i, reg in enumerate(unique_regions)}
    bar_colors = [region_color[r] for r in ordered_regions]

    # build plot
    values = [lang_total_kb[l] for l in ordered_langs]
    y_positions = list(range(len(ordered_langs)))

    # size heuristic: tall enough for labels
    fig_h = max(6, 0.28 * len(ordered_langs) + 1.5)
    plt.figure(figsize=(12, fig_h))

    plt.barh(y_positions, values, color=bar_colors)
    plt.yticks(y_positions, ordered_langs)
    plt.xlabel("Total size (KB)")
    plt.title("Language file sizes (KB), grouped by region and color-coded")

    # draw separators + region labels on the left
    # (keeps grouping obvious even with long lists)
    start = 0
    for reg in region_order:
        count = len(region_to_langs[reg])
        end = start + count
        if start > 0:
            plt.axhline(start - 0.5, linewidth=1)
        # annotate region near middle of its block
        mid = (start + end - 1) / 2
        plt.text(
            x=0,
            y=mid,
            s=f"  {reg}",
            va="center",
            ha="left",
            fontsize=10,
            alpha=0.9,
            transform=plt.gca().get_yaxis_transform(),  # x in axes coords, y in data coords
        )
        start = end

    # legend
    handles = []
    labels = []
    for reg in unique_regions:
        handles.append(plt.Line2D([0], [0], marker='s', linestyle='', color=region_color[reg]))
        labels.append(reg)
    plt.legend(handles, labels, title="Region", loc="lower right")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()

if __name__ == "__main__":
    main()


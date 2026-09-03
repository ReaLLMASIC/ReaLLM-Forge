#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# text_<lang>_<script>.txt
FNAME_RE = re.compile(r"^text_([a-z]{3})_([A-Za-z]{4})\.txt$")

def parse_size_to_kb(size_str: str) -> float:
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)(B?)\s*$", size_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"Bad size: {size_str}")
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "": 1/1024,
        "K": 1,
        "M": 1024,
        "G": 1024**2,
        "T": 1024**3,
        "P": 1024**4,
    }[unit]
    return val * mult

def script_to_region(script: str) -> str:
    mena = {"Arab", "Hebr"}
    south_asia = {"Deva", "Beng", "Gujr", "Guru", "Orya", "Knda", "Mlym"}
    east_asia = {"Jpan", "Hang", "Hani"}
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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="files.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-n", type=int, default=0)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # aggregate by (lang, script)
    ls_kb = Counter()
    ls_region = {}

    for r in rows:
        m = FNAME_RE.match(r["name"])
        if not m:
            continue
        lang, script = m.groups()
        key = f"{lang}_{script}"
        kb = parse_size_to_kb(r["size"])
        ls_kb[key] += kb
        ls_region[key] = script_to_region(script)

    # optionally limit to top-N
    items = ls_kb.most_common()
    if args.top_n > 0:
        items = items[:args.top_n]

    # group by region
    region_groups = defaultdict(list)
    for k, kb in items:
        region_groups[ls_region[k]].append((k, kb))

    # order regions by total size
    region_order = sorted(
        region_groups.keys(),
        key=lambda r: sum(v for _, v in region_groups[r]),
        reverse=True
    )

    ordered_keys = []
    ordered_vals = []
    ordered_regions = []

    for r in region_order:
        for k, kb in sorted(region_groups[r], key=lambda x: x[1], reverse=True):
            ordered_keys.append(k)
            ordered_vals.append(kb)
            ordered_regions.append(r)

    # color mapping (default matplotlib colors)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    region_color = {
        r: colors[i % len(colors)]
        for i, r in enumerate(region_order)
    }
    bar_colors = [region_color[r] for r in ordered_regions]

    # plot
    fig_h = max(6, 0.28 * len(ordered_keys) + 1.5)
    plt.figure(figsize=(12, fig_h))
    y = range(len(ordered_keys))
    plt.barh(y, ordered_vals, color=bar_colors)
    plt.yticks(y, ordered_keys)
    plt.xlabel("Total size (KB)")
    plt.title("Language–script file sizes (KB), grouped & colored by region")

    # region separators + labels
    start = 0
    for r in region_order:
        count = len(region_groups[r])
        if start > 0:
            plt.axhline(start - 0.5, linewidth=1)
        mid = start + (count - 1) / 2
        plt.text(
            0, mid, f"  {r}",
            va="center",
            ha="left",
            fontsize=10,
            transform=plt.gca().get_yaxis_transform()
        )
        start += count

    # legend
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=region_color[r])
        for r in region_order
    ]
    plt.legend(handles, region_order, title="Region", loc="lower right")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()

if __name__ == "__main__":
    main()


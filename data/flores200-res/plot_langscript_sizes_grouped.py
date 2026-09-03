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
        raise ValueError(f"Bad size: {size_str!r}")
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = {
        "": 1 / 1024,
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
    """
    Coarser “language grouping” / writing-system-family buckets.

    Examples requested:
      - Arab + Hebr + Ethi -> one group (Semitic scripts)
      - Hans + Hant (+Hani) -> one group (Han scripts)
    """
    semitic_scripts = {"Arab", "Hebr", "Ethi"}  # per your request (Amharic uses Ethiopic)
    han_scripts = {"Hans", "Hant", "Hani"}
    japanese = {"Jpan"}      # could fold into Han if you want, but keeping separate by default
    korean = {"Hang"}        # separate by default
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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="files.json", help="Input JSON listing")
    ap.add_argument("--out", default=None, help="Save figure to this path (png/pdf/etc)")
    ap.add_argument("--top-n", type=int, default=0, help="If >0, plot only top N entries by KB")
    ap.add_argument("--group-by", choices=["region", "script", "family"], default="region",
                    help="How to group entries on the Y axis (blocks)")
    ap.add_argument("--color-by", choices=["region", "script", "family"], default=None,
                    help="How to color bars. Default: same as --group-by")
    args = ap.parse_args()

    if args.color_by is None:
        args.color_by = args.group_by

    with open(args.json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # Aggregate by language-script key (no double counting across scripts)
    ls_kb = Counter()
    ls_script = {}
    ls_region = {}
    ls_family = {}

    for r in rows:
        m = FNAME_RE.match(r.get("name", ""))
        if not m:
            continue
        lang, script = m.groups()
        key = f"{lang}_{script}"
        kb = parse_size_to_kb(r["size"])

        ls_kb[key] += kb
        ls_script[key] = script
        ls_region[key] = script_to_region(script)
        ls_family[key] = script_to_family(script)

    items = ls_kb.most_common()
    if args.top_n and args.top_n > 0:
        items = items[:args.top_n]

    def get_label(key: str, which: str) -> str:
        if which == "region":
            return ls_region[key]
        if which == "script":
            return ls_script[key]
        if which == "family":
            return ls_family[key]
        raise ValueError(which)

    # Group and color labels
    groups = defaultdict(list)
    for key, kb in items:
        groups[get_label(key, args.group_by)].append((key, kb))

    group_order = sorted(groups.keys(), key=lambda g: sum(v for _, v in groups[g]), reverse=True)

    ordered_keys = []
    ordered_vals = []
    ordered_group_labels = []
    ordered_color_labels = []

    for g in group_order:
        for key, kb in sorted(groups[g], key=lambda x: x[1], reverse=True):
            ordered_keys.append(key)
            ordered_vals.append(kb)
            ordered_group_labels.append(g)
            ordered_color_labels.append(get_label(key, args.color_by))

    # Color map (use matplotlib default cycle)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not palette:
        palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    unique_color_labels = []
    for cl in ordered_color_labels:
        if cl not in unique_color_labels:
            unique_color_labels.append(cl)

    color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(unique_color_labels)}
    bar_colors = [color_map[cl] for cl in ordered_color_labels]

    # Plot
    fig_h = max(6, 0.28 * len(ordered_keys) + 1.5)
    plt.figure(figsize=(12, fig_h))

    y = range(len(ordered_keys))
    plt.barh(y, ordered_vals, color=bar_colors)
    plt.yticks(y, ordered_keys)
    plt.xlabel("Total size (KB)")
    plt.title(f"Language–script sizes (KB) | grouped by {args.group_by} | colored by {args.color_by}")

    # Group separators + labels
    start = 0
    for g in group_order:
        count = len(groups[g])
        if start > 0:
            plt.axhline(start - 0.5, linewidth=1)
        mid = start + (count - 1) / 2
        plt.text(
            0, mid, f"  {g}",
            va="center",
            ha="left",
            fontsize=10,
            transform=plt.gca().get_yaxis_transform()
        )
        start += count

    # Legend
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=color_map[cl]) for cl in unique_color_labels]
    plt.legend(handles, unique_color_labels, title=args.color_by, loc="lower right")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()

if __name__ == "__main__":
    main()


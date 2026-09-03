#!/usr/bin/env python3
"""Build toy simplified-Hanzi radical-location multicontext lanes.

This is intentionally small and transparent: it demonstrates a reversible
(1:1) representation by carrying the original simplified Hanzi in a dedicated
`char` lane and aligned radical-location lanes for model conditioning.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PLACEHOLDER = "∅"
NON_HANZI = "⧆"
LANES = ["char", "non_hanzi", "whole", "left", "right", "top", "bottom", "enclosure", "inside", "corner", "overlay", "other"]

# Demonstration lookup table: enough cases to cover the location categories and
# corner cases in input.txt. Values are radical/location signals, not full IDS.
DECOMP = {
    "一": {"whole":"一"}, "人": {"whole":"人"}, "口": {"whole":"口"},
    "明": {"left":"日", "right":"月"}, "休": {"left":"亻", "right":"木"},
    "林": {"left":"木", "right":"木"}, "好": {"left":"女", "right":"子"},
    "苗": {"top":"艹", "bottom":"田"}, "尖": {"top":"小", "bottom":"大"},
    "想": {"top":"相", "bottom":"心", "left":"木", "right":"目"},
    "国": {"enclosure":"囗", "inside":"玉"}, "问": {"enclosure":"门", "inside":"口"},
    "闪": {"enclosure":"门", "inside":"人"}, "医": {"enclosure":"匚", "inside":"矢"},
    "区": {"enclosure":"匚", "inside":"乂"}, "同": {"enclosure":"冂", "inside":"一口"},
    "这": {"enclosure":"辶", "inside":"文"}, "房": {"enclosure":"户", "inside":"方"},
    "病": {"enclosure":"疒", "inside":"丙"}, "氧": {"enclosure":"气", "inside":"羊"},
    "赢": {"corner":"亡口月贝凡"}, "器": {"corner":"口口口口", "inside":"犬"},
    "乘": {"overlay":"禾北"}, "爽": {"overlay":"大乂乂乂乂"},
    "坐": {"overlay":"人人土"}, "办": {"other":"力丶丶"}, "必": {"other":"心丿"},
}
# Tiny demo-only exclusions so the non-simplified-Hanzi vector is testable.
TRADITIONAL_ONLY = set("體龍門馬愛學國風書樂車東長萬與興貓鳥魚")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

def line_escape(value: str) -> str:
    """Encode one lane value so each timestep stays on one physical line."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

def is_simplified_hanzi(ch: str) -> bool:
    return len(ch) == 1 and bool(CJK_RE.fullmatch(ch)) and ch not in TRADITIONAL_ONLY

def encode_char(ch: str) -> dict[str, str]:
    if not is_simplified_hanzi(ch):
        row = {lane: NON_HANZI for lane in LANES}
        row["non_hanzi"] = ch
        return row
    row = {lane: PLACEHOLDER for lane in LANES}
    row["char"] = ch
    row["non_hanzi"] = PLACEHOLDER
    for lane, value in DECOMP.get(ch, {"other": ch}).items():
        row[lane] = value
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--output_root", default=".")
    ap.add_argument("--label", default="simplified_hanzi_mc")
    args = ap.parse_args()
    in_path = Path(args.input)
    out_root = Path(args.output_root)
    # Treat input.txt as an arbitrary UTF-8 text stream, not as one-character
    # records. This lets regular corpora such as poem/title files flow through
    # unchanged at the character timestep level: every code point gets one
    # aligned multicontext vector; non-Hanzi code points become NON_HANZI.
    chars = list(in_path.read_text(encoding="utf-8"))
    if not chars:
        raise ValueError(f"Input file is empty: {in_path}")
    rows = [encode_char(ch) for ch in chars]
    datasets = []
    for lane in LANES:
        lane_dir = out_root / lane
        lane_dir.mkdir(parents=True, exist_ok=True)
        lane_values = [line_escape(row[lane]) if lane == "non_hanzi" else row[lane] for row in rows]
        (lane_dir / "input.txt").write_text("\n".join(lane_values) + "\n", encoding="utf-8")
        datasets.append(f"simplified_hanzi_mc/{lane}/char_{args.label}")
    manifest = {"tokenizer":"simplified_hanzi_radical_location_multicontext", "source":str(in_path), "lanes":LANES,
                "multicontext_datasets":datasets, "placeholder":PLACEHOLDER, "non_hanzi":NON_HANZI,
                "bijection":"The char lane stores simplified Hanzi; the non_hanzi lane stores original non-Hanzi/non-simplified code points; aligned radical lanes store location labels.",
                "rows":len(rows)}
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()

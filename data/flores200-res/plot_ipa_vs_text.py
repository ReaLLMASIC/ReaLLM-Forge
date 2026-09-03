#!/usr/bin/env python3
"""
plot_ipa_vs_text.py

Graphs IPA vs raw text sizes for paired files across directories:

  text/<text_*.txt>
  ipa/<ipa_text_*.txt>

Defaults:
  --text-dir text/
  --ipa-dir  ipa/

Tokenization:
- Can also load tokenized sizes (e.g. tiktoken) from filtered_scripts.json (or other)
  produced by your tokenize_and_annotate_sizes.py pipeline, and plot:

    raw_bytes vs ipa_bytes vs tok_bytes

Assumptions for filtered JSON rows:
  - list[dict]
  - key "lang_script" OR ("language"+"_"+"script") matches the <lang> part of text_<lang>.txt
  - key "tokenized_sizes" is a dict like {"tiktoken": <KB float>, ...}

Produces (same as before):
- scatter: IPA bytes vs raw bytes
- bar: IPA/raw ratio
- bar: delta bytes (IPA - raw)

Additionally (if filtered json provided & matches are found):
- grouped bar: Raw vs IPA vs Tokenized (bytes) per language

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import math
import json

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class PairStats:
    lang: str
    raw_path: Path
    ipa_path: Path
    raw_bytes: int
    ipa_bytes: int
    raw_chars: int
    ipa_chars: int
    raw_lines: int
    ipa_lines: int
    tok_bytes: Optional[int] = None  # NEW

    @property
    def ratio_bytes(self) -> float:
        return (self.ipa_bytes / self.raw_bytes) if self.raw_bytes else float("inf")

    @property
    def delta_bytes(self) -> int:
        return self.ipa_bytes - self.raw_bytes


def read_stats(p: Path) -> Tuple[int, int, int]:
    """
    Returns (utf8_bytes, chars, lines).
    """
    data = p.read_text(encoding="utf-8", errors="replace")
    b = len(data.encode("utf-8"))
    c = len(data)
    lines = data.count("\n") + (1 if data and not data.endswith("\n") else 0)
    return b, c, lines


def discover_pairs(text_dir: Path, ipa_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Finds pairs across directories:
      text_dir/text_<lang>.txt
      ipa_dir/ipa_text_<lang>.txt

    Note: <lang> can be "eng_Latn" etc; we treat it as an opaque key.
    """
    raw_map: Dict[str, Path] = {}
    ipa_map: Dict[str, Path] = {}

    for p in text_dir.iterdir():
        if p.is_file() and p.name.startswith("text_") and p.name.endswith(".txt"):
            lang = p.name[len("text_") : -len(".txt")]
            raw_map[lang] = p

    for p in ipa_dir.iterdir():
        if p.is_file() and p.name.startswith("ipa_text_") and p.name.endswith(".txt"):
            lang = p.name[len("ipa_text_") : -len(".txt")]
            ipa_map[lang] = p

    langs = sorted(set(raw_map) & set(ipa_map))
    return [(lang, raw_map[lang], ipa_map[lang]) for lang in langs]


def _load_tokenized_kb_map(filtered_json: Path, method: str) -> Dict[str, float]:
    """
    Returns: { lang_script_key -> tokenized_size_kb } for the chosen method.

    Expects rows like:
      {
        "lang_script": "eng_Latn",
        "tokenized_sizes": {"tiktoken": 300.0},
        ...
      }
    """
    if not filtered_json.exists():
        raise FileNotFoundError(f"filtered json not found: {filtered_json}")

    rows = json.loads(filtered_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("filtered json must be a list of objects")

    out: Dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue

        key = r.get("lang_script")
        if not key:
            # try reconstruct
            lang = r.get("language")
            script = r.get("script")
            if lang and script:
                key = f"{lang}_{script}"

        if not key:
            continue

        tok_map = r.get("tokenized_sizes")
        if not isinstance(tok_map, dict):
            continue

        v = tok_map.get(method)
        if v is None:
            continue

        try:
            out[str(key)] = float(v)  # KB
        except Exception:
            continue

    return out


def make_scatter(stats: List[PairStats], outpath: Optional[Path], title: str) -> None:
    x = [s.raw_bytes for s in stats]
    y = [s.ipa_bytes for s in stats]
    labels = [s.lang for s in stats]

    plt.figure()
    plt.scatter(x, y)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4))

    plt.xlabel("Raw text size (UTF-8 bytes)")
    plt.ylabel("IPA text size (UTF-8 bytes)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def make_bar(
    stats: List[PairStats],
    values: List[float],
    ylabel: str,
    outpath: Optional[Path],
    title: str,
) -> None:
    langs = [s.lang for s in stats]
    plt.figure(figsize=(max(8, 0.8 * len(langs)), 5))
    plt.bar(langs, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    """
    Population mean/std (ddof=0) over vals.
    """
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def make_back_to_back_bar(
    stats: List[PairStats],
    outpath: Optional[Path],
    title: str = "Raw vs IPA Text Size (UTF-8 bytes)",
) -> None:
    """
    Back-to-back horizontal bar chart:
        - Raw text on the left (negative)
      - IPA text on the right (positive)

    Adds:
        - dotted mean lines for raw and ipa
      - dotted ±1 stddev lines for raw and ipa
    """
    langs = [s.lang for s in stats]
    raw_bytes = [float(s.raw_bytes) for s in stats]
    ipa_bytes = [float(s.ipa_bytes) for s in stats]

    raw_vals = [-b for b in raw_bytes]   # negative for left side
    ipa_vals = ipa_bytes                # positive for right side

    raw_mean, raw_std = _mean_std(raw_bytes)
    ipa_mean, ipa_std = _mean_std(ipa_bytes)

    y = range(len(langs))

    plt.figure(figsize=(10, max(5, 0.5 * len(langs))))
    plt.barh(y, raw_vals, label="Raw text", alpha=0.7)
    plt.barh(y, ipa_vals, label="IPA text", alpha=0.7)

    plt.yticks(y, langs)
    plt.axvline(0, color="black", linewidth=1)

    # Mean lines (dotted)
    plt.axvline(-raw_mean, linestyle=":", linewidth=2, label=f"Raw mean ({raw_mean:.0f})")
    plt.axvline(ipa_mean, linestyle=":", linewidth=2, label=f"IPA mean ({ipa_mean:.0f})")

    # ±1 stddev lines (dotted, lighter)
    plt.axvline(-(raw_mean - raw_std), linestyle=":", linewidth=1)
    plt.axvline(-(raw_mean + raw_std), linestyle=":", linewidth=1)
    plt.axvline(ipa_mean - ipa_std, linestyle=":", linewidth=1)
    plt.axvline(ipa_mean + ipa_std, linestyle=":", linewidth=1)

    plt.xlabel("UTF-8 bytes")
    plt.title(
        f"{title}\n"
        f"Raw mean={raw_mean:.0f}, std={raw_std:.0f} | "
        f"IPA mean={ipa_mean:.0f}, std={ipa_std:.0f}"
    )
    plt.grid(True, axis="x", linestyle="--", linewidth=0.5)

    max_val = max(max(ipa_vals), max(abs(v) for v in raw_vals))
    max_val = max(max_val, raw_mean + raw_std, ipa_mean + ipa_std)
    plt.xlim(-max_val * 1.15, max_val * 1.15)

    plt.legend(loc="best")

    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()

    print(f"[back-to-back] Raw bytes: mean={raw_mean:.2f}, std={raw_std:.2f}")
    print(f"[back-to-back] IPA bytes: mean={ipa_mean:.2f}, std={ipa_std:.2f}")


def make_grouped_raw_ipa_tok(
    stats: List[PairStats],
    outpath: Optional[Path],
    tok_label: str,
    title: str = "Raw vs IPA vs Tokenized Size (UTF-8 bytes)",
) -> None:
    """
    Grouped (clustered) vertical bar chart per language:
      raw, ipa, tok (if present)

    If some rows are missing tok_bytes, we simply omit that bar for that language.
    """
    langs = [s.lang for s in stats]
    raw = [s.raw_bytes for s in stats]
    ipa = [s.ipa_bytes for s in stats]
    tok = [s.tok_bytes for s in stats]  # Optional[int]

    x = list(range(len(langs)))
    width = 0.25

    plt.figure(figsize=(max(10, 0.9 * len(langs)), 5))

    # raw and ipa always present
    plt.bar([i - width for i in x], raw, width=width, label="Raw")
    plt.bar([i for i in x], ipa, width=width, label="IPA")

    # tokenized: only where present
    tok_x = []
    tok_y = []
    for i, v in enumerate(tok):
        if v is not None:
            tok_x.append(i + width)
            tok_y.append(v)
    if tok_x:
        plt.bar(tok_x, tok_y, width=width, label=tok_label)

    plt.xticks(x, langs, rotation=35, ha="right")
    plt.ylabel("Bytes (UTF-8)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def write_csv(stats: List[PairStats], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lang",
                "raw_path",
                "ipa_path",
                "raw_bytes",
                "ipa_bytes",
                "ratio_bytes",
                "delta_bytes",
                "raw_chars",
                "ipa_chars",
                "raw_lines",
                "ipa_lines",
                "tok_bytes",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.lang,
                    str(s.raw_path),
                    str(s.ipa_path),
                    s.raw_bytes,
                    s.ipa_bytes,
                    f"{s.ratio_bytes:.6f}",
                    s.delta_bytes,
                    s.raw_chars,
                    s.ipa_chars,
                    s.raw_lines,
                    s.ipa_lines,
                    "" if s.tok_bytes is None else s.tok_bytes,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph IPA vs raw text sizes across folders.")
    ap.add_argument("--text-dir", default="text", help="Directory with text_<lang>.txt files (default: text/)")
    ap.add_argument("--ipa-dir", default="ipa", help="Directory with ipa_text_<lang>.txt files (default: ipa/)")
    ap.add_argument("--save", action="store_true", help="Save plots instead of showing them.")
    ap.add_argument("--outdir", default="plots_out", help="Output directory when --save is used.")
    ap.add_argument("--csv", action="store_true", help="Also write CSV of statistics.")
    ap.add_argument(
        "--sort",
        choices=["lang", "raw_bytes", "ipa_bytes", "ratio", "delta"],
        default="lang",
        help="Sort order for plots.",
    )
    ap.add_argument(
        "--title",
        default="IPA vs Raw Text Size (UTF-8 bytes)",
        help="Title for scatter plot.",
    )

    # NEW: load tokenized sizes (KB) from filtered json and compare in bytes
    ap.add_argument(
        "--filtered-json",
        default=None,
        help="Optional: filtered_scripts.json (annotated) containing tokenized_sizes (KB). "
             "If set, we will add a Raw vs IPA vs Tokenized plot.",
    )
    ap.add_argument(
        "--tok-method",
        default="tiktoken",
        help="Which tokenized_sizes[method] to load from filtered json (default: tiktoken).",
    )
    ap.add_argument(
        "--skip-missing-tok",
        action="store_true",
        help="If set, drop languages that don't have tokenized_sizes[tok-method]. "
             "Default: keep language but omit tok bar.",
    )

    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    ipa_dir = Path(args.ipa_dir)

    if not text_dir.exists():
        raise SystemExit(f"text-dir not found: {text_dir}")
    if not ipa_dir.exists():
        raise SystemExit(f"ipa-dir not found: {ipa_dir}")

    pairs = discover_pairs(text_dir, ipa_dir)
    if not pairs:
        raise SystemExit("No matching text_/ipa_text_ pairs found.")

    tok_kb_map: Dict[str, float] = {}
    if args.filtered_json:
        tok_kb_map = _load_tokenized_kb_map(Path(args.filtered_json), method=args.tok_method)

    stats: List[PairStats] = []
    for lang, raw_p, ipa_p in pairs:
        rb, rc, rl = read_stats(raw_p)
        ib, ic, il = read_stats(ipa_p)

        tok_bytes: Optional[int] = None
        if tok_kb_map:
            kb = tok_kb_map.get(lang)
            if kb is not None:
                tok_bytes = int(round(kb * 1024.0))

        # optionally drop missing tokenized values
        if args.skip_missing_tok and tok_kb_map and tok_bytes is None:
            continue

        stats.append(
            PairStats(
                lang=lang,
                raw_path=raw_p,
                ipa_path=ipa_p,
                raw_bytes=rb,
                ipa_bytes=ib,
                raw_chars=rc,
                ipa_chars=ic,
                raw_lines=rl,
                ipa_lines=il,
                tok_bytes=tok_bytes,
            )
        )

    # Sorting
    key_map = {
        "lang": lambda s: s.lang,
        "raw_bytes": lambda s: s.raw_bytes,
        "ipa_bytes": lambda s: s.ipa_bytes,
        "ratio": lambda s: s.ratio_bytes,
        "delta": lambda s: s.delta_bytes,
    }
    stats.sort(key=key_map[args.sort])

    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    make_scatter(
        stats,
        outdir / "scatter_ipa_vs_raw_bytes.png" if args.save else None,
        args.title,
    )

    make_bar(
        stats,
        [s.ratio_bytes for s in stats],
        "IPA / Raw (bytes)",
        outdir / "bar_ratio_ipa_over_raw.png" if args.save else None,
        "IPA expansion ratio by language",
    )

    make_bar(
        stats,
        [float(s.delta_bytes) for s in stats],
        "IPA - Raw (bytes)",
        outdir / "bar_delta_ipa_minus_raw.png" if args.save else None,
        "Absolute size increase (IPA − Raw)",
    )

    make_back_to_back_bar(
        stats,
        outdir / "bar_back_to_back_raw_vs_ipa.png" if args.save else None,
        title="Raw vs IPA Text Size by Language (UTF-8 bytes)",
    )

    # NEW: grouped raw vs ipa vs tokenized (if filtered_json provided and any matches exist)
    if tok_kb_map:
        any_tok = any(s.tok_bytes is not None for s in stats)
        if any_tok:
            make_grouped_raw_ipa_tok(
                stats,
                outdir / f"bar_grouped_raw_ipa_{args.tok_method}.png" if args.save else None,
                tok_label=args.tok_method,
                title=f"Raw vs IPA vs {args.tok_method} (bytes)",
            )
        else:
            print(f"[warn] --filtered-json provided but no tokenized_sizes['{args.tok_method}'] matched your lang keys.")

    if args.save and args.csv:
        write_csv(stats, outdir / "ipa_vs_raw_stats.csv")

    for s in stats:
        tok_str = "n/a" if s.tok_bytes is None else str(s.tok_bytes)
        print(
            f"{s.lang:14s} raw={s.raw_bytes:8d} "
            f"ipa={s.ipa_bytes:8d} "
            f"tok({args.tok_method})={tok_str:>8s} "
            f"ratio={s.ratio_bytes:6.3f} "
            f"delta={s.delta_bytes:8d}"
        )


if __name__ == "__main__":
    main()


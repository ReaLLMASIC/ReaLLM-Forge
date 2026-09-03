#!/usr/bin/env python3
"""Build restructured FLORES-200 text files and run stepped char-BPE sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_URL = "https://huggingface.co/datasets/muhammadravi251001/restructured-flores200/tree/main/data"

LANGUAGES = [
    ("kiswahili", "swh_Latn"),
    ("bahasa_indonesian", "ind_Latn"),
    ("korean", "kor_Hang"),
    ("english", "eng_Latn"),
    ("chinese", "zho_Hans"),
    ("japanese", "jpn_Jpan"),
    ("arabic", "arb_Arab"),
    ("spanish", "spa_Latn"),
    ("german", "deu_Latn"),
    ("russian", "rus_Cyrl"),
    ("thai", "tha_Thai"),
    ("filipino", "tgl_Latn"),
    ("hindi", "hin_Deva"),
    ("finnish", "fin_Latn"),
    ("italian", "ita_Latn"),
]

DEFAULT_VOCAB_SIZES = [320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_vocab_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    bad = [size for size in sizes if size <= 256]
    if bad:
        raise argparse.ArgumentTypeError(
            f"char_bpe vocab sizes must be > 256 because IDs 0..255 are byte fallback tokens: {bad}"
        )
    return sizes


def emit_restructured_flores_text(code: str, output_path: Path, source_url: str) -> None:
    """Emit one text_<code> column from the restructured FLORES-200 parquet files.

    This mirrors data/flores200-res/get_dataset.sh, which uses the shared
    data/template/utils/get_parquet_dataset.py helper against the
    muhammadravi251001/restructured-flores200 Hugging Face parquet folder.
    """
    helper = repo_root() / "data" / "template" / "utils" / "get_parquet_dataset.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(helper),
            "--url",
            source_url,
            "--include_keys",
            f"text_{code}",
            "--value_prefix",
            "\n",
            "--output_text_file",
            str(output_path),
        ],
        check=True,
        cwd=output_path.parent,
    )


def write_language_texts(text_dir: Path, refresh: bool, source_url: str) -> dict[str, Path]:
    text_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for label, code in LANGUAGES:
        out_path = text_dir / f"{label}.txt"
        if refresh or not out_path.exists():
            emit_restructured_flores_text(code, out_path, source_url)
        outputs[label] = out_path

    korean_nfd = text_dir / "korean_nfd.txt"
    if refresh or not korean_nfd.exists():
        converter = repo_root() / "data" / "hangul" / "hangul_nfc_to_nfd.py"
        subprocess.run(
            [sys.executable, str(converter), str(outputs["korean"]), str(korean_nfd)],
            check=True,
            cwd=repo_root(),
        )
    outputs["korean_nfd"] = korean_nfd
    return outputs


def count_bin_tokens(path: Path, vocab_size: int) -> int:
    dtype_bytes = 4 if vocab_size > 65535 else 2
    return path.stat().st_size // dtype_bytes


def run_prepare(
    label: str,
    text_path: Path,
    vocab_size: int,
    run_dir: Path,
    percentage_train: float,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    for artifact in (
        "train.bin",
        "val.bin",
        "meta.pkl",
        "char_bpe_vocab.json",
        "char_bpe_token_counts.json",
        "metrics.json",
    ):
        (run_dir / artifact).unlink(missing_ok=True)
    prepare = repo_root() / "data" / "template" / "prepare.py"
    cmd = [
        sys.executable,
        str(prepare),
        "--train_input",
        str(text_path),
        "--method",
        "char_bpe",
        "--vocab_size",
        str(vocab_size),
        "--percentage_train",
        str(percentage_train),
        "--track_token_counts",
        "--train_output",
        "train.bin",
        "--val_output",
        "val.bin",
    ]
    subprocess.run(cmd, check=True, cwd=run_dir)

    meta_path = run_dir / "meta.pkl"
    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    train_tokens = count_bin_tokens(run_dir / "train.bin", meta["vocab_size"])
    val_path = run_dir / "val.bin"
    val_tokens = count_bin_tokens(val_path, meta["vocab_size"]) if val_path.exists() else 0
    text = text_path.read_text(encoding="utf-8")
    split_index = int(len(text) * percentage_train)
    train_text = text[:split_index]
    val_text = text[split_index:] if percentage_train < 1.0 else ""
    byte_fallback_tokens = sum(
        count for token_id, count in meta.get("token_counts", {}).items() if int(token_id) < 256
    )

    metrics = {
        "language": label,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": meta["vocab_size"],
        "utf8_bytes": len(text.encode("utf-8")),
        "unicode_codepoints": len(text),
        "train_utf8_bytes": len(train_text.encode("utf-8")),
        "val_utf8_bytes": len(val_text.encode("utf-8")),
        "train_unicode_codepoints": len(train_text),
        "val_unicode_codepoints": len(val_text),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "bytes_per_token": round(len(train_text.encode("utf-8")) / train_tokens, 6) if train_tokens else 0,
        "val_bytes_per_token": round(len(val_text.encode("utf-8")) / val_tokens, 6) if val_tokens else 0,
        "chars_per_token": round(len(train_text) / train_tokens, 6) if train_tokens else 0,
        "val_chars_per_token": round(len(val_text) / val_tokens, 6) if val_tokens else 0,
        "unk_byte_fallback_tokens": byte_fallback_tokens,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def write_summary(rows: list[dict[str, object]], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "summary.json"
    csv_path = results_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = repo_root() / "data" / "char_bpe_exploration"
    parser.add_argument("--base-dir", type=Path, default=base)
    parser.add_argument("--vocab-sizes", type=parse_vocab_sizes, default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--refresh-texts", action="store_true")
    parser.add_argument("--clean-runs", action="store_true")
    parser.add_argument(
        "--source-url",
        default=SOURCE_URL,
        help="Hugging Face parquet folder URL for muhammadravi251001/restructured-flores200.",
    )
    parser.add_argument(
        "--percentage-train",
        type=float,
        default=0.9,
        help="Fraction of each language text used for train.bin; the remainder becomes val.bin.",
    )
    args = parser.parse_args()

    if not 0 < args.percentage_train <= 1:
        raise SystemExit("--percentage-train must be in the interval (0, 1].")

    if args.clean_runs:
        shutil.rmtree(args.base_dir / "runs", ignore_errors=True)
        shutil.rmtree(args.base_dir / "results", ignore_errors=True)

    texts = write_language_texts(args.base_dir / "texts", args.refresh_texts, args.source_url)
    ordered_labels = [label for label, _ in LANGUAGES] + ["korean_nfd"]

    rows: list[dict[str, object]] = []
    for label in ordered_labels:
        for vocab_size in args.vocab_sizes:
            run_dir = args.base_dir / "runs" / label / f"vocab_{vocab_size}"
            print(f"[char_bpe_exploration] {label} vocab={vocab_size}", flush=True)
            rows.append(run_prepare(label, texts[label], vocab_size, run_dir, args.percentage_train))
            write_summary(rows, args.base_dir / "results")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

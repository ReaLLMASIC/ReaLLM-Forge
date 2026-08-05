#!/usr/bin/env python3
from __future__ import annotations
import argparse
from array import array
import pickle
from pathlib import Path

from extract_jamo_lite_streams import LANE_NAMES, lite_features


def dtype_for_vocab(vocab_size: int) -> str:
    return "uint32" if vocab_size > 65535 else "uint16"


def write_ids(path: Path, ids: list[int], dtype: str) -> None:
    values = array("I" if dtype == "uint32" else "H", ids)
    with path.open("wb") as f:
        values.tofile(f)


def encode_text(text: str, meta_path: Path) -> tuple[list[int], str]:
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    stoi = meta.get("stoi")
    if stoi is None:
        raise ValueError(f"Only char-tokenizer meta.pkl files are supported, missing stoi in {meta_path}")
    missing = sorted({ch for ch in text if ch not in stoi})
    if missing:
        formatted = ", ".join(repr(ch) for ch in missing[:10])
        raise ValueError(f"Prompt contains characters absent from {meta_path}: {formatted}")
    return [stoi[ch] for ch in text], dtype_for_vocab(int(meta.get("vocab_size", len(stoi))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a rendered prompt into Korean jamo-lite multicontext start files.")
    parser.add_argument("prompt")
    parser.add_argument("output_dir")
    parser.add_argument("--dataset-root", default="data/korean_jamo_mc")
    args = parser.parse_args()

    lane_buffers = {lane: [] for lane in LANE_NAMES}
    for ch in args.prompt:
        values = dict(zip(LANE_NAMES, lite_features(ch)))
        for lane, value in values.items():
            lane_buffers[lane].append(value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    dtypes: set[str] = set()
    start_files: list[str] = []
    for lane in LANE_NAMES:
        text = "".join(lane_buffers[lane])
        ids, dtype = encode_text(text, dataset_root / lane / "meta.pkl")
        dtypes.add(dtype)
        out_path = output_dir / f"{lane}.bin"
        write_ids(out_path, ids, dtype)
        start_files.append(str(out_path))

    if len(dtypes) != 1:
        raise ValueError(f"Mixed prompt dtypes are not supported by sample.py: {', '.join(sorted(dtypes))}")
    dtype = next(iter(dtypes))
    (output_dir / "start_files.txt").write_text("\n".join(start_files) + "\n", encoding="utf-8")
    (output_dir / "dtype.txt").write_text(dtype + "\n", encoding="utf-8")
    print(dtype)


if __name__ == "__main__":
    main()

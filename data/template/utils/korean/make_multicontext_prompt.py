#!/usr/bin/env python3
from __future__ import annotations
import argparse
from array import array
import pickle
from pathlib import Path

from hangul_factorizer import HangulFactorizedTokenizer


def _dtype_for_vocab(vocab_size: int) -> str:
    return "uint32" if vocab_size > 65535 else "uint16"


def _write_ids(path: Path, ids: list[int], dtype: str) -> None:
    typecode = "I" if dtype == "uint32" else "H"
    values = array(typecode, ids)
    with path.open("wb") as f:
        values.tofile(f)


def _encode_with_meta(text: str, meta_path: Path) -> tuple[list[int], str]:
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    stoi = meta.get("stoi")
    if stoi is None:
        raise ValueError(f"Only char-tokenizer meta.pkl files are supported, missing stoi in {meta_path}")
    missing = sorted({ch for ch in text if ch not in stoi})
    if missing:
        formatted = ", ".join(repr(ch) for ch in missing[:10])
        raise ValueError(f"Prompt contains characters absent from {meta_path}: {formatted}")
    vocab_size = int(meta.get("vocab_size", len(stoi)))
    return [stoi[ch] for ch in text], _dtype_for_vocab(vocab_size)


def main() -> None:
    p = argparse.ArgumentParser(description="Encode a rendered prompt into per-lane multicontext .bin start files.")
    p.add_argument("prompt", help="Rendered prompt text, e.g. 'English: Hello Korean: '")
    p.add_argument("output_dir", help="Directory for <lane>.bin prompt files")
    p.add_argument("--dataset-root", default="data/korean_mc", help="Dataset root containing lane meta.pkl files")
    p.add_argument("--use-pos", "--pos", action="store_true", help="Include part-of-speech (POS) tag lane using kiwipiepy")
    args = p.parse_args()

    tok = HangulFactorizedTokenizer(use_pos=args.use_pos)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lane_text = {name: [] for name in tok.lane_names}
    char_text: list[str] = []
    if tok.use_pos:
        for item in tok.encode_text(args.prompt):
            ch = item["char"]
            ids = item["indices"]
            for i, idx in enumerate(ids):
                lane_text[tok.lane_names[i]].append(tok.token_for(i, idx))
            char_text.append(ch)
    else:
        for ch in args.prompt:
            ids = tok.encode_char(ch)
            for i, idx in enumerate(ids):
                lane_text[tok.lane_names[i]].append(tok.token_for(i, idx))
            char_text.append(ch)

    start_files: list[str] = []
    dtypes: set[str] = set()
    for name in [*tok.lane_names, "char"]:
        text = "".join(char_text if name == "char" else lane_text[name])
        ids, dtype = _encode_with_meta(text, dataset_root / name / "meta.pkl")
        dtypes.add(dtype)
        out_path = output_dir / f"{name}.bin"
        _write_ids(out_path, ids, dtype)
        start_files.append(str(out_path))

    if len(dtypes) != 1:
        names = ", ".join(sorted(dtypes))
        raise ValueError(f"Mixed prompt dtypes are not supported by sample.py --multicontext_start_file_dtype: {names}")
    dtype_name = next(iter(dtypes))
    (output_dir / "start_files.txt").write_text("\n".join(start_files) + "\n", encoding="utf-8")
    (output_dir / "dtype.txt").write_text(dtype_name + "\n", encoding="utf-8")
    print(dtype_name)


if __name__ == "__main__":
    main()

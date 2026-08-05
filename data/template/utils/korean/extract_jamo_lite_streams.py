#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import TextIO

S_BASE = 0xAC00
L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT
SPECIAL_PAD = "_"
LANE_NAMES = ["char", "first_jamo", "last_jamo", "eun_neun"]


def is_modern_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and S_BASE <= ord(ch) < S_BASE + S_COUNT


def decompose_indices(ch: str) -> tuple[int, int, int]:
    s_index = ord(ch) - S_BASE
    return s_index // N_COUNT, (s_index % N_COUNT) // T_COUNT, s_index % T_COUNT


def lite_features(ch: str) -> tuple[str, str, str, str]:
    if not is_modern_hangul_syllable(ch):
        return ch, SPECIAL_PAD, SPECIAL_PAD, SPECIAL_PAD
    l_index, v_index, t_index = decompose_indices(ch)
    first = chr(L_BASE + l_index)
    last = chr(T_BASE + t_index) if t_index else chr(V_BASE + v_index)
    particle = "은" if t_index else "는"
    return ch, first, last, particle


def stream_extract(input_path: Path, output_dir: Path, *, chunk_size: int, buffering: int, metadata_json: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, TextIO] = {}
    metadata_fp: TextIO | None = None
    first_record = True
    position = 0
    try:
        for lane in LANE_NAMES:
            lane_dir = output_dir / lane
            lane_dir.mkdir(parents=True, exist_ok=True)
            handles[lane] = (lane_dir / "input.txt").open("w", encoding="utf-8", buffering=buffering)
        if metadata_json:
            metadata_fp = (output_dir / metadata_json).open("w", encoding="utf-8", buffering=buffering)
            metadata_fp.write('{\n  "lanes": ')
            json.dump(LANE_NAMES, metadata_fp, ensure_ascii=False)
            metadata_fp.write(',\n  "characters": [\n')

        with input_path.open("r", encoding="utf-8", errors="replace", buffering=buffering) as in_fp:
            while True:
                chunk = in_fp.read(chunk_size)
                if not chunk:
                    break
                buffers = {lane: [] for lane in LANE_NAMES}
                for ch in chunk:
                    char, first, last, particle = lite_features(ch)
                    values = {"char": char, "first_jamo": first, "last_jamo": last, "eun_neun": particle}
                    for lane, value in values.items():
                        buffers[lane].append(value)
                    if metadata_fp is not None:
                        if not first_record:
                            metadata_fp.write(",\n")
                        json.dump({"position": position, "char": ch, "is_hangul_nfc": is_modern_hangul_syllable(ch), "lanes": values}, metadata_fp, ensure_ascii=False)
                        first_record = False
                    position += 1
                for lane, chars in buffers.items():
                    handles[lane].write("".join(chars))
    finally:
        if metadata_fp is not None:
            metadata_fp.write("\n  ]\n}\n")
            metadata_fp.close()
        for handle in handles.values():
            handle.close()
    (output_dir / "lane_metadata.json").write_text(json.dumps({"lanes": LANE_NAMES, "num_characters": position, "special_pad": SPECIAL_PAD}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lightweight Korean jamo multicontext lanes.")
    parser.add_argument("input", help="UTF-8 Korean text file")
    parser.add_argument("output_dir", help="Output directory containing char/ and special lane directories")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--buffering", type=int, default=1024 * 1024)
    parser.add_argument("--metadata-json", default="metadata.json", help="Metadata sidecar relative to output_dir; use '' to disable")
    args = parser.parse_args()
    stream_extract(Path(args.input), Path(args.output_dir), chunk_size=args.chunk_size, buffering=args.buffering, metadata_json=args.metadata_json)


if __name__ == "__main__":
    main()

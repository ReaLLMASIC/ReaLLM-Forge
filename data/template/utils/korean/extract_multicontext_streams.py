#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import TextIO

from hangul_factorizer import HangulFactorizedTokenizer, dump_json


def _json_dump(obj: object, fp: TextIO, *, indent: int | None = None) -> None:
    json.dump(obj, fp, ensure_ascii=False, indent=indent)


def _write_yaml_header(fp: TextIO, tok: HangulFactorizedTokenizer) -> None:
    fp.write("lanes:\n")
    for lane in tok.lane_metadata():
        fp.write(f"- index: {lane['index']}\n")
        fp.write(f"  name: {json.dumps(lane['name'], ensure_ascii=False)}\n")
        fp.write(f"  description: {json.dumps(lane['description'], ensure_ascii=False)}\n")
        fp.write("  values:\n")
        for value in lane["values"]:
            fp.write(f"  - {json.dumps(value, ensure_ascii=False)}\n")
    fp.write("characters:\n")


def _write_yaml_character(fp: TextIO, record: dict) -> None:
    fp.write(f"- position: {record['position']}\n")
    fp.write(f"  char: {json.dumps(record['char'], ensure_ascii=False)}\n")
    fp.write(f"  codepoint: {json.dumps(record['codepoint'], ensure_ascii=False)}\n")
    fp.write(f"  is_hangul: {str(record['is_hangul']).lower()}\n")
    fp.write("  lanes:\n")
    for lane_name, lane_record in record["lanes"].items():
        fp.write(f"    {lane_name}:\n")
        fp.write(f"      id: {lane_record['id']}\n")
        fp.write(f"      value: {json.dumps(lane_record['value'], ensure_ascii=False)}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Stream Korean text into aligned 23- or 24-lane Hangul factor streams.")
    p.add_argument("input", help="UTF-8 Korean text file")
    p.add_argument("output_dir", help="Directory that will receive lane/input.txt files")
    p.add_argument("--metadata-json", default="metadata.json", help="Streaming JSON sidecar path relative to output_dir; use '' to disable")
    p.add_argument("--metadata-yaml", default="metadata.yaml", help="Streaming YAML sidecar path relative to output_dir; use '' to disable")
    p.add_argument("--buffering", type=int, default=1024 * 1024, help="Per-file output buffer size in bytes")
    p.add_argument("--chunk-size", type=int, default=65536, help="Input characters to process before flushing bounded lane buffers")
    p.add_argument("--use-pos", "--pos", action="store_true", help="Include part-of-speech (POS) tag lane using kiwipiepy")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = HangulFactorizedTokenizer(use_pos=args.use_pos)

    lane_handles: list[TextIO] = []
    json_fp: TextIO | None = None
    yaml_fp: TextIO | None = None
    first_json_record = True
    position = 0

    try:
        for name in tok.lane_names:
            lane_dir = out / name
            lane_dir.mkdir(parents=True, exist_ok=True)
            lane_handles.append((lane_dir / "input.txt").open("w", encoding="utf-8", buffering=args.buffering))

        char_dir = out / "char"
        char_dir.mkdir(parents=True, exist_ok=True)
        char_fp = (char_dir / "input.txt").open("w", encoding="utf-8", buffering=args.buffering)
        lane_handles.append(char_fp)

        if args.metadata_json:
            json_fp = (out / args.metadata_json).open("w", encoding="utf-8", buffering=args.buffering)
            json_fp.write('{\n  "lanes": ')
            _json_dump(tok.lane_metadata(), json_fp, indent=2)
            json_fp.write(',\n  "characters": [\n')
        if args.metadata_yaml:
            yaml_fp = (out / args.metadata_yaml).open("w", encoding="utf-8", buffering=args.buffering)
            _write_yaml_header(yaml_fp, tok)

        with Path(args.input).open("r", encoding="utf-8", errors="replace", buffering=args.buffering) as input_fp:
            while True:
                chunk = input_fp.read(args.chunk_size)
                if not chunk:
                    break
                chunk_lane_chars = [[] for _ in tok.lane_names]
                if tok.use_pos:
                    encoded_seq = tok.encode_text(chunk)
                    for item in encoded_seq:
                        ch = item["char"]
                        ids = item["indices"]
                        for i, idx in enumerate(ids):
                            chunk_lane_chars[i].append(tok.token_for(i, idx))

                        if json_fp is not None or yaml_fp is not None:
                            pos_tag = tok.id_to_value[23][ids[23]] if len(ids) > 23 else "UNK"
                            record = tok.metadata_for_char(ch, position, pos_tag=pos_tag)
                            if json_fp is not None:
                                if not first_json_record:
                                    json_fp.write(",\n")
                                json_fp.write("    ")
                                _json_dump(record, json_fp)
                                first_json_record = False
                            if yaml_fp is not None:
                                _write_yaml_character(yaml_fp, record)
                        position += 1
                else:
                    for ch in chunk:
                        ids = tok.encode_char(ch)
                        for i, idx in enumerate(ids):
                            chunk_lane_chars[i].append(tok.token_for(i, idx))

                        if json_fp is not None or yaml_fp is not None:
                            record = tok.metadata_for_char(ch, position)
                            if json_fp is not None:
                                if not first_json_record:
                                    json_fp.write(",\n")
                                json_fp.write("    ")
                                _json_dump(record, json_fp)
                                first_json_record = False
                            if yaml_fp is not None:
                                _write_yaml_character(yaml_fp, record)
                        position += 1

                for i, chars in enumerate(chunk_lane_chars):
                    lane_handles[i].write("".join(chars))
                char_fp.write(chunk)
    finally:
        if json_fp is not None:
            json_fp.write("\n  ]\n}\n")
            json_fp.close()
        if yaml_fp is not None:
            yaml_fp.close()
        for handle in lane_handles:
            handle.close()

    dump_json(out / "lane_metadata.json", {"lanes": tok.lane_metadata(), "num_characters": position})

if __name__ == "__main__":
    main()

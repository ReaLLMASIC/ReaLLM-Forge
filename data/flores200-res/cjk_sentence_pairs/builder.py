from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE = "flores200-res"
RAW_TO_CANONICAL = {
    "kor_Hang": "ko",
    "zho_Hans": "zh",
    "jpn_Jpan": "ja",
}
RAW_INPUT_ORDER = ("kor_Hang", "zho_Hans", "jpn_Jpan")
CANONICAL_LANGUAGE_ORDER = [RAW_TO_CANONICAL[code] for code in RAW_INPUT_ORDER]


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    records: int
    skipped: dict[str, int]
    input_paths: dict[str, Path]

    @property
    def unordered_sentence_pairs(self) -> int:
        return self.records * 3

    @property
    def directed_sentence_pairs(self) -> int:
        return self.records * 6


def _resolve_input_paths(input_path: Path) -> dict[str, Path]:
    base_dir = input_path if input_path.is_dir() else input_path.parent
    paths = {code: base_dir / f"text_{code}.txt" for code in RAW_INPUT_ORDER}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required FLORES input file(s): " + ", ".join(missing)
        )
    return paths


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _record_id(translations: dict[str, str]) -> str:
    payload = json.dumps(
        translations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_records(
    lines_by_code: dict[str, list[str]],
) -> tuple[list[dict[str, object]], Counter[str]]:
    skipped: Counter[str] = Counter()
    max_len = max(len(lines) for lines in lines_by_code.values())
    seen_ids: set[str] = set()
    records: list[dict[str, object]] = []

    for line_number in range(1, max_len + 1):
        row: dict[str, str] = {}
        missing_language = False

        for raw_code in RAW_INPUT_ORDER:
            lines = lines_by_code[raw_code]
            if line_number > len(lines):
                missing_language = True
                continue
            row[RAW_TO_CANONICAL[raw_code]] = lines[line_number - 1].strip()

        if missing_language:
            skipped["missing_parallel_sentence"] += 1
            continue

        if any(not row[lang] for lang in CANONICAL_LANGUAGE_ORDER):
            skipped["empty_sentence"] += 1
            continue

        record_id = _record_id(row)
        if record_id in seen_ids:
            skipped["duplicate_aligned_record"] += 1
            continue
        seen_ids.add(record_id)

        records.append(
            {
                "id": record_id,
                "line_number": line_number,
                "translations": {lang: row[lang] for lang in CANONICAL_LANGUAGE_ORDER},
            }
        )

    records.sort(key=lambda record: (record["id"], record["line_number"]))
    return records, skipped


def build_sentence_pairs(input_path: Path | str, output_path: Path | str) -> BuildResult:
    input_paths = _resolve_input_paths(Path(input_path))
    lines_by_code = {code: _read_lines(path) for code, path in input_paths.items()}
    records, skipped = _iter_records(lines_by_code)

    payload = {
        "source": SOURCE,
        "languages": CANONICAL_LANGUAGE_ORDER,
        "records": records,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return BuildResult(
        output_path=out_path,
        records=len(records),
        skipped=dict(sorted(skipped.items())),
        input_paths=input_paths,
    )


def _format_skipped(skipped: dict[str, int]) -> str:
    if not skipped:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in skipped.items())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic CJK sentence translation records from FLORES files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one FLORES text file or to the directory containing all CJK files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path, for example data/cjk_sentence_pairs.json.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = build_sentence_pairs(args.input, args.out)
    print(f"output={result.output_path}")
    print(f"records={result.records}")
    print(f"unordered_sentence_pairs={result.unordered_sentence_pairs}")
    print(f"directed_sentence_pairs={result.directed_sentence_pairs}")
    print(f"skipped={_format_skipped(result.skipped)}")
    return 0

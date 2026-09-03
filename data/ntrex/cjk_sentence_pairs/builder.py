from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE = "ntrex"
TAG_TO_LANGUAGE = {
    "KO": "kor_Hang",
    "ZH": "zho_Hans",
    "JA": "jpn_Jpan",
}
CANONICAL_TO_LANGUAGE = {
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
}
LANGUAGE_ORDER = ["kor_Hang", "zho_Hans", "jpn_Jpan"]


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    records: int
    skipped: dict[str, int]
    input_path: Path

    @property
    def unordered_sentence_pairs(self) -> int:
        return self.records * 3

    @property
    def directed_sentence_pairs(self) -> int:
        return self.records * 6


def _record_id(translations: dict[str, str]) -> str:
    payload = json.dumps(
        translations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_sentence(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _parse_tag(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("#") and stripped.endswith(":"):
        return stripped[1:-1]
    return None


def _iter_aligned_rows(input_path: Path) -> tuple[list[dict[str, object]], Counter[str]]:
    skipped: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    current_tag: str | None = None
    current_parts: list[str] = []
    row: dict[str, str] = {}
    row_start_line: int | None = None

    def flush_sentence() -> None:
        nonlocal current_tag, current_parts
        if current_tag in TAG_TO_LANGUAGE:
            row[TAG_TO_LANGUAGE[current_tag]] = _normalize_sentence(current_parts)
        current_tag = None
        current_parts = []

    def flush_row(next_row_start: int | None = None) -> None:
        nonlocal row, row_start_line
        if not row:
            row_start_line = next_row_start
            return

        missing = [language for language in LANGUAGE_ORDER if language not in row]
        if missing:
            skipped["missing_cjk_language"] += 1
        elif any(not row[language] for language in LANGUAGE_ORDER):
            skipped["empty_sentence"] += 1
        else:
            translations = {language: row[language] for language in LANGUAGE_ORDER}
            record_id = _record_id(translations)
            if record_id in seen_ids:
                skipped["duplicate_aligned_record"] += 1
            else:
                seen_ids.add(record_id)
                records.append(
                    {
                        "id": record_id,
                        "line_number": row_start_line,
                        "translations": translations,
                    }
                )

        row = {}
        row_start_line = next_row_start

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            tag = _parse_tag(line)
            if tag is None:
                if current_tag is not None:
                    current_parts.append(line)
                elif line.strip():
                    skipped["text_outside_language_block"] += 1
                continue

            flush_sentence()
            if tag == "EN":
                flush_row(next_row_start=line_number)
            elif tag not in TAG_TO_LANGUAGE:
                skipped["unsupported_language_block"] += 1
            current_tag = tag
            current_parts = []

    flush_sentence()
    flush_row()
    records.sort(key=lambda record: (record["id"], record["line_number"]))
    return records, skipped


def build_sentence_pairs(input_path: Path | str, output_path: Path | str) -> BuildResult:
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    records, skipped = _iter_aligned_rows(in_path)
    payload = {
        "source": SOURCE,
        "languages": LANGUAGE_ORDER,
        "canonical_language_codes": sorted(CANONICAL_TO_LANGUAGE),
        "language_aliases": CANONICAL_TO_LANGUAGE,
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
        input_path=in_path,
    )


def _format_skipped(skipped: dict[str, int]) -> str:
    if not skipped:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in skipped.items())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic CJK sentence translation records from NTREX input.txt."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to NTREX input.txt containing #EN/#KO/#ZH/#JA aligned blocks.",
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

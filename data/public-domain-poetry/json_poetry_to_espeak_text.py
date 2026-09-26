#!/usr/bin/env python3
"""Create an eSpeak-safe text corpus from poetry JSON or JSONL.

Each accepted record contributes its Author, Title, and text fields. Common
typographic punctuation is converted to ASCII before validation. By default,
the whole record is skipped if any field still contains a character outside:

  * printable ASCII (U+0020 through U+007E)
  * tab and newline
  * Unicode Latin letters (for example, ae ligatures and accented letters)

Examples:

  python3 json_poetry_to_espeak_text.py poems.json -o poems_espeak.txt

  python3 json_poetry_to_espeak_text.py poems.jsonl -o poems_espeak.txt \
      --invalid-policy replace --reject-report rejected.jsonl \
      --character-report emitted_characters.json

File output is split into 10 MB shards by default, named poems_espeak_0001.txt,
poems_espeak_0002.txt, and so on. Change this with --max-output-size; set it to
0 to disable sharding and write the exact --output filename.

Use --ascii-only to transliterate common Latin letters and strip diacritics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, BinaryIO, Iterable, Iterator, TextIO


# Punctuation and spacing which commonly occur in public-domain text. Mapping
# these before validation keeps otherwise clean poems without asking eSpeak to
# interpret layout characters or inconsistent Unicode punctuation.
TYPOGRAPHIC_TRANSLATIONS = str.maketrans(
    {
        "\u00a0": " ",   # no-break space
        "\u1680": " ",   # ogham space mark
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
        "\u2010": "-",   # hyphen
        "\u2011": "-",   # non-breaking hyphen
        "\u2012": "-",   # figure dash
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2015": "-",   # horizontal bar
        "\u2212": "-",   # minus sign
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2026": "...", # ellipsis
        "\u00ad": "",    # soft hyphen
        "\ufeff": "",    # byte-order mark / zero-width no-break space
    }
)


# Characters which do not decompose into ASCII with NFKD.
ASCII_LATIN_TRANSLITERATIONS = str.maketrans(
    {
        "\u00c6": "AE", "\u00e6": "ae",
        "\u0152": "OE", "\u0153": "oe",
        "\u00d0": "D",  "\u00f0": "d",
        "\u00de": "Th", "\u00fe": "th",
        "\u00df": "ss", "\u1e9e": "SS",
        "\u0141": "L",  "\u0142": "l",
        "\u00d8": "O",  "\u00f8": "o",
        "\u0110": "D",  "\u0111": "d",
        "\u0126": "H",  "\u0127": "h",
        "\u0131": "i",
    }
)


def is_allowed_character(character: str, *, ascii_only: bool) -> bool:
    """Return whether one normalized character is safe for this corpus."""
    if character in {"\n", "\t"}:
        return True
    codepoint = ord(character)
    if 0x20 <= codepoint <= 0x7E:
        return True
    if ascii_only:
        return False
    return unicodedata.category(character).startswith("L") and "LATIN" in unicodedata.name(
        character, ""
    )


def to_ascii_latin(text: str) -> str:
    """Transliterate Latin text to ASCII without a third-party dependency."""
    text = text.translate(ASCII_LATIN_TRANSLITERATIONS)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def normalize_and_validate(
    value: str,
    *,
    ascii_only: bool,
    invalid_policy: str,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Normalize a field and return it together with invalid-char details."""
    original_value = value
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.translate(TYPOGRAPHIC_TRANSLATIONS)
    value = unicodedata.normalize("NFC", value)
    if ascii_only:
        value = to_ascii_latin(value)

    invalid: dict[str, dict[str, Any]] = {}
    output: list[str] = []

    for offset, character in enumerate(value):
        if is_allowed_character(character, ascii_only=ascii_only):
            output.append(character)
            continue

        detail = invalid.setdefault(
            character,
            {
                "character": character,
                "codepoint": f"U+{ord(character):04X}",
                "unicode_name": unicodedata.name(character, "UNNAMED"),
                "count": 0,
                "first_offset": offset,
            },
        )
        detail["count"] += 1
        if invalid_policy == "replace":
            # A space avoids accidentally joining words on either side of an
            # unsupported character. Repeated spaces are cleaned below.
            output.append(" ")

    normalized = "".join(output)
    if invalid_policy == "replace":
        normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = normalized.strip()
    return normalized, list(invalid.values()), normalized != original_value


def iter_jsonl(stream: Iterable[str], source_name: str) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source_name}:{line_number}: invalid JSONL: {error.msg}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(
                f"{source_name}:{line_number}: expected a JSON object, "
                f"found {type(item).__name__}"
            )
        yield item


def iter_records(input_path: str) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array/object or a JSONL file."""
    if input_path == "-":
        source_name = "<stdin>"
        stream_context = nullcontext(sys.stdin)
        suffix = ""
    else:
        source_name = input_path
        stream_context = Path(input_path).open("r", encoding="utf-8-sig")
        suffix = Path(input_path).suffix.lower()

    with stream_context as stream:
        if suffix in {".jsonl", ".ndjson"}:
            yield from iter_jsonl(stream, source_name)
            return

        contents = stream.read()
        try:
            data = json.loads(contents)
        except json.JSONDecodeError as array_error:
            # Some downloaded files have a .json extension but contain JSONL.
            try:
                yield from iter_jsonl(contents.splitlines(), source_name)
                return
            except (ValueError, TypeError) as jsonl_error:
                raise ValueError(
                    f"{source_name}: neither valid JSON nor valid JSONL "
                    f"({array_error.msg})"
                ) from jsonl_error

        if isinstance(data, dict):
            yield data
            return
        if not isinstance(data, list):
            raise ValueError(
                f"{source_name}: top-level JSON must be an object or array of objects"
            )
        for item_number, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{source_name}: item {item_number} is "
                    f"{type(item).__name__}, not an object"
                )
            yield item


def format_record(author: str, title: str, text: str, *, labels: bool) -> str:
    if labels:
        return f"#Author:\n{author}\n#Title:\n{title}\n#Text:\n{text}"
    return f"{author}\n{title}\n{text}"


SIZE_UNITS = {
    "": 1,
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "KIB": 1_024,
    "MIB": 1_048_576,
    "GIB": 1_073_741_824,
}


def parse_byte_size(value: str) -> int:
    """Parse values such as 500000, 10MB, or 10MiB into bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            "size must be a byte count or a value such as 10MB or 10MiB"
        )
    number = float(match.group(1))
    unit = match.group(2).upper()
    if unit not in SIZE_UNITS:
        raise argparse.ArgumentTypeError(
            f"unknown size unit {match.group(2)!r}; use B, KB, MB, GB, KiB, MiB, or GiB"
        )
    byte_count = int(number * SIZE_UNITS[unit])
    if number > 0 and byte_count == 0:
        raise argparse.ArgumentTypeError("size is too small to contain one byte")
    return byte_count


def indexed_output_path(base_path: Path, index: int) -> Path:
    """Insert a one-based, zero-padded shard index before the final suffix."""
    return base_path.with_name(f"{base_path.stem}_{index:04d}{base_path.suffix}")


def utf8_prefix_at_most(data: bytes, limit: int) -> int:
    """Return a positive UTF-8 boundary no larger than limit."""
    if len(data) <= limit:
        return len(data)
    cut = limit
    # If data[cut] is a continuation byte, cut points inside a multibyte code
    # point. Back up to its leading byte so the prefix remains valid UTF-8.
    while cut > 0 and data[cut] & 0xC0 == 0x80:
        cut -= 1
    if cut == 0:
        raise ValueError("maximum output size is too small for one UTF-8 character")
    return cut


class ShardedCorpusWriter:
    """Write record blocks to byte-limited, UTF-8-valid output shards."""

    def __init__(self, base_path: Path, max_bytes: int) -> None:
        self.base_path = base_path
        self.max_bytes = max_bytes
        self.output_paths: list[Path] = []
        self.bytes_written = 0
        self._stream: BinaryIO | None = None
        self._current_size = 0

    def __enter__(self) -> "ShardedCorpusWriter":
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _open_next(self) -> None:
        if self._stream is not None:
            self._stream.close()
        index = len(self.output_paths) + 1
        path = (
            indexed_output_path(self.base_path, index)
            if self.max_bytes > 0
            else self.base_path
        )
        self._stream = path.open("wb")
        self.output_paths.append(path)
        self._current_size = 0

    def _rotate(self) -> None:
        self._open_next()

    def _write_bytes(self, data: bytes) -> None:
        while data:
            if self._stream is None:
                self._open_next()

            if self.max_bytes == 0:
                assert self._stream is not None
                self._stream.write(data)
                self._current_size += len(data)
                self.bytes_written += len(data)
                return

            remaining = self.max_bytes - self._current_size
            if remaining == 0:
                self._rotate()
                remaining = self.max_bytes

            cut = utf8_prefix_at_most(data, remaining)
            if cut < len(data):
                # Prefer a human-readable split after whitespace when doing so
                # still uses at least half of the available shard space.
                minimum_preferred = max(1, cut // 2)
                newline = data.rfind(b"\n", minimum_preferred, cut)
                space = data.rfind(b" ", minimum_preferred, cut)
                preferred = max(newline, space)
                if preferred >= minimum_preferred:
                    cut = preferred + 1

            assert self._stream is not None
            chunk = data[:cut]
            self._stream.write(chunk)
            self._current_size += len(chunk)
            self.bytes_written += len(chunk)
            data = data[cut:]
            if data:
                self._rotate()

    def write_record(self, record: str) -> None:
        # A trailing blank line separates records and makes each normal shard
        # independently suitable as eSpeak input.
        block = (record + "\n\n").encode("utf-8")

        if self.max_bytes > 0 and self._current_size:
            # Preserve a normal-sized record as a unit. Oversized records start
            # in a fresh shard and are then split safely by _write_bytes().
            if len(block) > self.max_bytes or self._current_size + len(block) > self.max_bytes:
                self._rotate()
        self._write_bytes(block)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class StreamCorpusWriter:
    """Unsharded writer used for stdout."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.output_paths: list[Path] = []
        self.bytes_written = 0
        self._first_record = True

    def __enter__(self) -> "StreamCorpusWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._first_record:
            self.stream.write("\n")
            self.bytes_written += 1

    def write_record(self, record: str) -> None:
        prefix = "" if self._first_record else "\n\n"
        output = prefix + record
        self.stream.write(output)
        self.bytes_written += len(output.encode("utf-8"))
        self._first_record = False


def write_character_report(path: Path, counts: Counter[str]) -> None:
    report = [
        {
            "character": character,
            "escaped": character.encode("unicode_escape").decode("ascii"),
            "codepoint": f"U+{ord(character):04X}",
            "unicode_name": unicodedata.name(character, "UNNAMED"),
            "count": count,
        }
        for character, count in sorted(counts.items(), key=lambda item: ord(item[0]))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate Author, Title, and text from JSON/JSONL records while "
            "keeping only records suitable for an eSpeak text corpus."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="JSON/JSONL input file(s); use - to read one input from stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="poems_espeak.txt",
        help=(
            "base output filename, or - for unsharded stdout; indexed shard "
            "names are inserted before its suffix (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-output-size",
        "--max-size",
        type=parse_byte_size,
        default=10_000_000,
        metavar="SIZE",
        help=(
            "maximum UTF-8 byte size per output shard, accepting values such "
            "as 10MB or 10MiB; 0 disables sharding (default: 10MB)"
        ),
    )
    parser.add_argument("--author-key", default="Author")
    parser.add_argument("--title-key", default="Title")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--invalid-policy",
        choices=("skip-record", "replace"),
        default="skip-record",
        help=(
            "skip a record containing unsupported characters, or replace each "
            "unsupported character with a space (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="transliterate Latin letters to ASCII and strip diacritics",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="omit the #Author, #Title, and #Text labels",
    )
    parser.add_argument(
        "--reject-report",
        type=Path,
        help="optional JSONL report describing every rejected record",
    )
    parser.add_argument(
        "--character-report",
        type=Path,
        help="optional JSON inventory of characters written to the output",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.inputs.count("-") > 1:
        raise ValueError("stdin (-) may be specified only once")

    if args.output == "-":
        output_context: ShardedCorpusWriter | StreamCorpusWriter = StreamCorpusWriter(
            sys.stdout
        )
    else:
        output_path = Path(args.output)
        output_context = ShardedCorpusWriter(output_path, args.max_output_size)

    if args.reject_report:
        args.reject_report.parent.mkdir(parents=True, exist_ok=True)
        reject_context = args.reject_report.open("w", encoding="utf-8", newline="\n")
    else:
        reject_context = nullcontext(None)

    totals = Counter()
    emitted_characters: Counter[str] = Counter()

    with output_context as output, reject_context as reject_output:
        for input_path in args.inputs:
            for source_index, item in enumerate(iter_records(input_path), start=1):
                totals["records_seen"] += 1
                fields: dict[str, str] = {}
                invalid_by_field: dict[str, list[dict[str, Any]]] = {}
                missing_fields: list[str] = []
                record_was_cleaned = False

                for output_name, input_key in (
                    ("author", args.author_key),
                    ("title", args.title_key),
                    ("text", args.text_key),
                ):
                    raw_value = item.get(input_key)
                    if not isinstance(raw_value, str) or not raw_value.strip():
                        missing_fields.append(input_key)
                        continue
                    normalized, invalid, field_was_cleaned = normalize_and_validate(
                        raw_value,
                        ascii_only=args.ascii_only,
                        invalid_policy=args.invalid_policy,
                    )
                    record_was_cleaned = record_was_cleaned or field_was_cleaned
                    if not normalized:
                        missing_fields.append(input_key)
                    fields[output_name] = normalized
                    if invalid:
                        invalid_by_field[input_key] = invalid

                rejection_reason: str | None = None
                if missing_fields:
                    rejection_reason = "missing_or_empty_field"
                    totals["rejected_missing_or_empty"] += 1
                elif invalid_by_field and args.invalid_policy == "skip-record":
                    rejection_reason = "unsupported_character"
                    totals["rejected_unsupported_character"] += 1

                if rejection_reason:
                    if reject_output is not None:
                        json.dump(
                            {
                                "source": input_path,
                                "record_number": source_index,
                                "reason": rejection_reason,
                                "missing_fields": missing_fields,
                                "invalid_characters": invalid_by_field,
                                "author": str(item.get(args.author_key, ""))[:160],
                                "title": str(item.get(args.title_key, ""))[:160],
                            },
                            reject_output,
                            ensure_ascii=False,
                        )
                        reject_output.write("\n")
                    continue

                record = format_record(
                    fields["author"],
                    fields["title"],
                    fields["text"],
                    labels=not args.no_labels,
                )
                output.write_record(record)
                emitted_characters.update(record)
                totals["records_written"] += 1
                if record_was_cleaned:
                    totals["records_cleaned"] += 1

    if args.character_report:
        write_character_report(args.character_report, emitted_characters)

    summary = (
        f"Seen: {totals['records_seen']}; written: {totals['records_written']}; "
        f"cleaned: {totals['records_cleaned']}; rejected (missing/empty): "
        f"{totals['rejected_missing_or_empty']}; rejected (unsupported): "
        f"{totals['rejected_unsupported_character']}; emitted characters: "
        f"{len(emitted_characters)}; output bytes: {output.bytes_written}; "
        f"output files: {len(output.output_paths) if args.output != '-' else 1}"
    )
    print(summary, file=sys.stderr)
    if output.output_paths:
        if len(output.output_paths) == 1:
            print(f"Output: {output.output_paths[0]}", file=sys.stderr)
        else:
            print(
                f"Outputs: {output.output_paths[0]} through {output.output_paths[-1]}",
                file=sys.stderr,
            )
    return 0


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())



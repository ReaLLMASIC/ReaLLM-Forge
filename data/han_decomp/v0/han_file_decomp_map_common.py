#!/usr/bin/env python3
"""
Shared helpers for reversible Han decomposition file maps.

The map format is intentionally self-contained:
  * exact original file bytes can be embedded once at the document level;
  * each token preserves its original source span and editable current text;
  * Han-main-block tokens optionally link to a full character inventory entry
    plus a compact per-token decomposition summary.

The implementation is scoped to UTF encodings because the caller explicitly
asked for UTF text and because exact per-token byte-span recovery is far more
reliable for UTF-8/16/32 than for stateful legacy encodings.
"""
from __future__ import annotations

import base64
import bisect
import copy
import dataclasses
import datetime as _dt
import hashlib
import json
import sys
import unicodedata
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "han-file-map/v1"
MAIN_BLOCK_START = 0x4E00
MAIN_BLOCK_END = 0x9FFF
MAIN_BLOCK_NAME = "CJK Unified Ideographs"
DEFAULT_COMPONENT_SEPARATOR = " "
DEFAULT_TOKEN_SEPARATOR = ""
DEFAULT_PAYLOAD_CODEC = "zlib+base64"
SUPPORTED_MAP_FORMATS = {"json", "jsonl"}
SUPPORTED_PAYLOAD_CODECS = {"base64", "zlib+base64"}
SUPPORTED_UTF_ENCODINGS = {
    "utf-8",
    "utf8",
    "utf-8-sig",
    "utf8-sig",
    "utf-16",
    "utf16",
    "utf-16-le",
    "utf16le",
    "utf-16-be",
    "utf16be",
    "utf-32",
    "utf32",
    "utf-32-le",
    "utf32le",
    "utf-32-be",
    "utf32be",
}
UTF8_BOM = b"\xef\xbb\xbf"
UTF16_LE_BOM = b"\xff\xfe"
UTF16_BE_BOM = b"\xfe\xff"
UTF32_LE_BOM = b"\xff\xfe\x00\x00"
UTF32_BE_BOM = b"\x00\x00\xfe\xff"


class HanFileMapError(Exception):
    """Base exception for reversible file-map helpers."""


@dataclasses.dataclass(frozen=True)
class UTFEncodingPlan:
    requested_encoding: str
    normalized_requested_encoding: str
    decode_encoding: str
    encode_encoding: str
    errors: str
    bom_bytes: bytes

    def to_dict(self) -> dict:
        return {
            "requested_encoding": self.requested_encoding,
            "normalized_requested_encoding": self.normalized_requested_encoding,
            "decode_encoding": self.decode_encoding,
            "encode_encoding": self.encode_encoding,
            "errors": self.errors,
            "bom_bytes_b64": b64encode_bytes(self.bom_bytes),
            "bom_length": len(self.bom_bytes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UTFEncodingPlan":
        try:
            return cls(
                requested_encoding=data["requested_encoding"],
                normalized_requested_encoding=data["normalized_requested_encoding"],
                decode_encoding=data["decode_encoding"],
                encode_encoding=data["encode_encoding"],
                errors=data.get("errors", "strict"),
                bom_bytes=b64decode_bytes(data.get("bom_bytes_b64", "")),
            )
        except KeyError as exc:
            raise HanFileMapError(f"Malformed encoding plan in map: missing {exc}") from exc


@dataclasses.dataclass(frozen=True)
class Position:
    line: int
    column: int


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def b64encode_bytes(data: bytes) -> str:
    if not data:
        return ""
    return base64.b64encode(data).decode("ascii")


def b64decode_bytes(data: str) -> bytes:
    if not data:
        return b""
    return base64.b64decode(data.encode("ascii"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text_utf8(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_encoding_label(label: str) -> str:
    return label.strip().lower().replace("_", "-")


def is_main_block_char(ch: str) -> bool:
    return len(ch) == 1 and MAIN_BLOCK_START <= ord(ch) <= MAIN_BLOCK_END


def cp_to_uplus(cp: int) -> str:
    if cp <= 0xFFFF:
        return f"U+{cp:04X}"
    return f"U+{cp:06X}"


def char_codepoints(text: str) -> List[str]:
    return [cp_to_uplus(ord(ch)) for ch in text]


def detect_utf_plan(raw_bytes: bytes, encoding: str, errors: str) -> UTFEncodingPlan:
    norm = normalize_encoding_label(encoding)
    if norm not in SUPPORTED_UTF_ENCODINGS:
        raise HanFileMapError(
            "This mapper only supports UTF encodings for exact reversible byte spans. "
            f"Got: {encoding!r}. Use UTF-8/16/32 or transcode first."
        )

    if norm in {"utf-8", "utf8"}:
        return UTFEncodingPlan(encoding, norm, "utf-8", "utf-8", errors, b"")

    if norm in {"utf-8-sig", "utf8-sig"}:
        bom = UTF8_BOM if raw_bytes.startswith(UTF8_BOM) else b""
        return UTFEncodingPlan(encoding, norm, "utf-8-sig", "utf-8", errors, bom)

    if norm in {"utf-16", "utf16"}:
        if raw_bytes.startswith(UTF16_LE_BOM):
            return UTFEncodingPlan(encoding, norm, "utf-16", "utf-16-le", errors, UTF16_LE_BOM)
        if raw_bytes.startswith(UTF16_BE_BOM):
            return UTFEncodingPlan(encoding, norm, "utf-16", "utf-16-be", errors, UTF16_BE_BOM)
        endian = "utf-16-le" if sys.byteorder == "little" else "utf-16-be"
        return UTFEncodingPlan(encoding, norm, "utf-16", endian, errors, b"")

    if norm in {"utf-16-le", "utf16le"}:
        return UTFEncodingPlan(encoding, norm, "utf-16-le", "utf-16-le", errors, b"")
    if norm in {"utf-16-be", "utf16be"}:
        return UTFEncodingPlan(encoding, norm, "utf-16-be", "utf-16-be", errors, b"")

    if norm in {"utf-32", "utf32"}:
        if raw_bytes.startswith(UTF32_LE_BOM):
            return UTFEncodingPlan(encoding, norm, "utf-32", "utf-32-le", errors, UTF32_LE_BOM)
        if raw_bytes.startswith(UTF32_BE_BOM):
            return UTFEncodingPlan(encoding, norm, "utf-32", "utf-32-be", errors, UTF32_BE_BOM)
        endian = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        return UTFEncodingPlan(encoding, norm, "utf-32", endian, errors, b"")

    if norm in {"utf-32-le", "utf32le"}:
        return UTFEncodingPlan(encoding, norm, "utf-32-le", "utf-32-le", errors, b"")
    if norm in {"utf-32-be", "utf32be"}:
        return UTFEncodingPlan(encoding, norm, "utf-32-be", "utf-32-be", errors, b"")

    raise HanFileMapError(f"Unsupported UTF encoding plan: {encoding!r}")


def decode_text(raw_bytes: bytes, plan: UTFEncodingPlan) -> str:
    try:
        return raw_bytes.decode(plan.decode_encoding, plan.errors)
    except UnicodeError as exc:
        raise HanFileMapError(
            f"Failed to decode input bytes using {plan.decode_encoding!r} with errors={plan.errors!r}: {exc}"
        ) from exc


def encode_text(text: str, plan: UTFEncodingPlan, *, preserve_bom: bool = True, errors: Optional[str] = None) -> bytes:
    err = plan.errors if errors is None else errors
    try:
        encoded = text.encode(plan.encode_encoding, err)
    except UnicodeError as exc:
        raise HanFileMapError(
            f"Failed to encode text using {plan.encode_encoding!r} with errors={err!r}: {exc}"
        ) from exc
    if preserve_bom and plan.bom_bytes:
        return plan.bom_bytes + encoded
    return encoded


def encode_payload(raw_bytes: bytes, codec: str) -> dict:
    if codec not in SUPPORTED_PAYLOAD_CODECS:
        raise HanFileMapError(f"Unsupported payload codec: {codec}")
    if codec == "base64":
        stored = raw_bytes
    else:
        stored = zlib.compress(raw_bytes, level=9)
    return {
        "codec": codec,
        "size_bytes": len(raw_bytes),
        "sha256_bytes": sha256_bytes(raw_bytes),
        "data_b64": b64encode_bytes(stored),
    }


def decode_payload(payload: Optional[dict]) -> Optional[bytes]:
    if payload is None:
        return None
    codec = payload.get("codec")
    data_b64 = payload.get("data_b64")
    if codec is None or data_b64 is None:
        raise HanFileMapError("Malformed payload block in mapped file")
    data = b64decode_bytes(data_b64)
    if codec == "base64":
        raw = data
    elif codec == "zlib+base64":
        try:
            raw = zlib.decompress(data)
        except zlib.error as exc:
            raise HanFileMapError(f"Failed to decompress payload: {exc}") from exc
    else:
        raise HanFileMapError(f"Unsupported payload codec in map: {codec}")
    expected_hash = payload.get("sha256_bytes")
    if expected_hash and sha256_bytes(raw) != expected_hash:
        raise HanFileMapError("Embedded original payload hash mismatch")
    return raw


def count_newline_kinds(text: str) -> Dict[str, int]:
    counts = {"CRLF": 0, "LF": 0, "CR": 0}
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\r":
            if i + 1 < len(text) and text[i + 1] == "\n":
                counts["CRLF"] += 1
                i += 2
                continue
            counts["CR"] += 1
        elif ch == "\n":
            counts["LF"] += 1
        i += 1
    return counts


def build_line_start_offsets(text: str) -> List[int]:
    starts = [0]
    offset = 0
    if not text:
        return starts
    for segment in text.splitlines(keepends=True):
        offset += len(segment)
        starts.append(offset)
    if starts[-1] != len(text):
        starts.append(len(text))
    return starts


def offset_to_line_col(offset: int, line_starts: Sequence[int]) -> Position:
    if offset < 0:
        raise HanFileMapError(f"Negative text offset: {offset}")
    idx = bisect.bisect_right(line_starts, offset) - 1
    if idx < 0:
        idx = 0
    if idx >= len(line_starts):
        idx = len(line_starts) - 1
    return Position(line=idx + 1, column=(offset - line_starts[idx]) + 1)


def classify_char(ch: str) -> str:
    if ch in {"\r", "\n"}:
        return "newline"
    if is_main_block_char(ch):
        return "han_main_block"
    if ch.isspace():
        return "whitespace"
    return "other"


def unicode_name_safe(ch: str) -> Optional[str]:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return None


def deep_copy_jsonable(value):
    return copy.deepcopy(value)


def iter_decomposition_records(dataset_path: Path) -> Iterator[dict]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise HanFileMapError(f"Expected top-level list in decomposition dataset: {dataset_path}")
        for item in data:
            yield item
        return
    raise HanFileMapError(f"Unsupported decomposition dataset format: {dataset_path}")


def load_decomposition_subset(dataset_path: Path, wanted_chars: Iterable[str]) -> Dict[str, dict]:
    wanted = set(wanted_chars)
    out: Dict[str, dict] = {}
    if not wanted:
        return out
    for record in iter_decomposition_records(dataset_path):
        ch = record.get("char")
        if ch in wanted and ch not in out:
            out[ch] = record
            if len(out) == len(wanted):
                break
    return out


def build_decomposition_summary(char_record: Optional[dict], component_separator: str) -> Optional[dict]:
    if not char_record:
        return None
    decomp = char_record.get("decomposition") or {}
    unihan = char_record.get("unihan") or {}
    raw_leaf = list(decomp.get("leaf_components_raw") or [])
    normalized_leaf = list(decomp.get("leaf_radicals_normalized") or [])
    return {
        "available": True,
        "primary_raw": decomp.get("primary_raw"),
        "immediate_components": list(decomp.get("immediate_components") or []),
        "leaf_components_raw": raw_leaf,
        "leaf_radicals_normalized": normalized_leaf,
        "leaf_components_raw_unique": list(decomp.get("leaf_components_raw_unique") or []),
        "leaf_radicals_normalized_unique": list(decomp.get("leaf_radicals_normalized_unique") or []),
        "unresolved_tokens": list(decomp.get("unresolved_tokens") or []),
        "expanded_tree": deep_copy_jsonable(decomp.get("expanded_tree")),
        "render_raw_default": component_separator.join(raw_leaf),
        "render_normalized_default": component_separator.join(normalized_leaf),
        "kRSUnicode": deep_copy_jsonable(unihan.get("kRSUnicode")),
        "kTotalStrokes": deep_copy_jsonable(unihan.get("kTotalStrokes")),
        "kDefinition": unihan.get("kDefinition"),
    }


def make_inventory(
    chars_in_order: Sequence[str],
    occurrence_counts: Dict[str, int],
    record_lookup: Dict[str, dict],
) -> Tuple[List[dict], Dict[str, int]]:
    inventory: List[dict] = []
    index_by_char: Dict[str, int] = {}
    for ch in chars_in_order:
        if ch in index_by_char:
            continue
        idx = len(inventory)
        index_by_char[ch] = idx
        inventory.append(
            {
                "inventory_index": idx,
                "char": ch,
                "occurrence_count": occurrence_counts.get(ch, 0),
                "record_available": ch in record_lookup,
                "char_record": deep_copy_jsonable(record_lookup.get(ch)),
            }
        )
    return inventory, index_by_char


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_token_records(
    text: str,
    plan: UTFEncodingPlan,
    record_lookup: Dict[str, dict],
    component_separator: str,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    line_starts = build_line_start_offsets(text)
    han_chars_in_text = [ch for ch in text if is_main_block_char(ch)]
    unique_han_first_order = dedupe_preserve_order(han_chars_in_text)
    occurrence_counts: Dict[str, int] = {}
    for ch in han_chars_in_text:
        occurrence_counts[ch] = occurrence_counts.get(ch, 0) + 1
    inventory, inventory_index = make_inventory(unique_han_first_order, occurrence_counts, record_lookup)

    tokens: List[dict] = []
    byte_offset = len(plan.bom_bytes)
    for token_id, ch in enumerate(text):
        char_start = token_id
        char_end = token_id + 1
        encoded = ch.encode(plan.encode_encoding, plan.errors)
        byte_start = byte_offset
        byte_end = byte_start + len(encoded)
        byte_offset = byte_end
        start_pos = offset_to_line_col(char_start, line_starts)
        end_pos = offset_to_line_col(char_end, line_starts)
        kind = classify_char(ch)
        inv_idx = inventory_index.get(ch) if is_main_block_char(ch) else None
        char_record = record_lookup.get(ch) if is_main_block_char(ch) else None
        token = {
            "token_id": token_id,
            "kind": kind,
            "text_original": ch,
            "text_current": ch,
            "original_length_codepoints": len(ch),
            "codepoints": char_codepoints(ch),
            "codepoint_ints": [ord(cp) for cp in ch],
            "unicode_name": unicode_name_safe(ch),
            "general_category": unicodedata.category(ch),
            "inventory_index": inv_idx,
            "positions": {
                "char_start": char_start,
                "char_end": char_end,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "line_start": start_pos.line,
                "column_start": start_pos.column,
                "line_end": end_pos.line,
                "column_end": end_pos.column,
            },
            "decomposition": build_decomposition_summary(char_record, component_separator) if inv_idx is not None else None,
            "annotations": {},
        }
        tokens.append(token)
    return tokens, inventory, inventory_index


def reconstruct_token_text(
    token: dict,
    *,
    mode: str,
    component_separator: str,
    preserve_non_han: bool = True,
    missing_han_fallback: str = "current",
) -> str:
    if mode == "original-text":
        return token.get("text_original", "")
    if mode == "current-text":
        return token.get("text_current", token.get("text_original", ""))

    if mode not in {"decomp-raw", "decomp-normalized"}:
        raise HanFileMapError(f"Unsupported reconstruction mode: {mode}")

    if token.get("kind") != "han_main_block":
        if preserve_non_han:
            return token.get("text_current", token.get("text_original", ""))
        return ""

    decomp = token.get("decomposition") or {}
    if mode == "decomp-raw":
        seq = list(decomp.get("leaf_components_raw") or [])
    else:
        seq = list(decomp.get("leaf_radicals_normalized") or [])

    if seq:
        return component_separator.join(seq)

    if missing_han_fallback == "current":
        return token.get("text_current", token.get("text_original", ""))
    if missing_han_fallback == "original":
        return token.get("text_original", "")
    if missing_han_fallback == "empty":
        return ""
    raise HanFileMapError(f"Unsupported missing Han fallback: {missing_han_fallback}")


def reconstruct_text_from_tokens(
    tokens: Sequence[dict],
    *,
    mode: str,
    component_separator: str,
    token_separator: str = DEFAULT_TOKEN_SEPARATOR,
    preserve_non_han: bool = True,
    missing_han_fallback: str = "current",
) -> str:
    rendered = [
        reconstruct_token_text(
            token,
            mode=mode,
            component_separator=component_separator,
            preserve_non_han=preserve_non_han,
            missing_han_fallback=missing_han_fallback,
        )
        for token in tokens
    ]
    return token_separator.join(rendered)


def first_difference(a: bytes, b: bytes) -> Optional[dict]:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return {
                "offset": idx,
                "a_byte_hex": f"{a[idx]:02X}",
                "b_byte_hex": f"{b[idx]:02X}",
            }
    if len(a) != len(b):
        return {
            "offset": limit,
            "a_byte_hex": f"{a[limit]:02X}" if len(a) > limit else None,
            "b_byte_hex": f"{b[limit]:02X}" if len(b) > limit else None,
        }
    return None


def changed_tokens_for_mode(
    tokens: Sequence[dict],
    *,
    mode: str,
    component_separator: str,
    preserve_non_han: bool = True,
    missing_han_fallback: str = "current",
) -> List[dict]:
    out: List[dict] = []
    for token in tokens:
        original = token.get("text_original", "")
        rendered = reconstruct_token_text(
            token,
            mode=mode,
            component_separator=component_separator,
            preserve_non_han=preserve_non_han,
            missing_han_fallback=missing_han_fallback,
        )
        if rendered != original:
            out.append(
                {
                    "token_id": token.get("token_id"),
                    "kind": token.get("kind"),
                    "positions": deep_copy_jsonable(token.get("positions")),
                    "text_original": original,
                    "rendered_text": rendered,
                    "text_current": token.get("text_current", original),
                    "inventory_index": token.get("inventory_index"),
                    "codepoints": list(token.get("codepoints") or []),
                }
            )
    return out


def current_text_edits(tokens: Sequence[dict]) -> List[dict]:
    edits: List[dict] = []
    for token in tokens:
        original = token.get("text_original", "")
        current = token.get("text_current", original)
        if current != original:
            edits.append(
                {
                    "token_id": token.get("token_id"),
                    "kind": token.get("kind"),
                    "positions": deep_copy_jsonable(token.get("positions")),
                    "text_original": original,
                    "text_current": current,
                    "inventory_index": token.get("inventory_index"),
                    "codepoints": list(token.get("codepoints") or []),
                }
            )
    return edits


def write_map_document(document: dict, output_path: Path, output_format: str) -> None:
    if output_format not in SUPPORTED_MAP_FORMATS:
        raise HanFileMapError(f"Unsupported map output format: {output_format}")
    ensure_parent_dir(output_path)
    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(document, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return

    doc_header = {k: v for k, v in document.items() if k not in {"inventory", "tokens"}}
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"record_type": "document", **doc_header}, ensure_ascii=False))
        fh.write("\n")
        for inv in document.get("inventory", []):
            fh.write(json.dumps({"record_type": "inventory", **inv}, ensure_ascii=False))
            fh.write("\n")
        for token in document.get("tokens", []):
            fh.write(json.dumps({"record_type": "token", **token}, ensure_ascii=False))
            fh.write("\n")


def load_map_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        validate_map_document(doc)
        return doc

    if suffix == ".jsonl":
        header = None
        inventory: List[dict] = []
        tokens: List[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                rtype = record.pop("record_type", None)
                if rtype == "document":
                    header = record
                elif rtype == "inventory":
                    inventory.append(record)
                elif rtype == "token":
                    tokens.append(record)
                else:
                    raise HanFileMapError(f"Unknown JSONL map record_type: {rtype}")
        if header is None:
            raise HanFileMapError(f"Missing document header in JSONL map: {path}")
        doc = dict(header)
        doc["inventory"] = inventory
        doc["tokens"] = tokens
        validate_map_document(doc)
        return doc

    raise HanFileMapError(f"Unsupported mapped file format: {path}")


def validate_map_document(document: dict) -> None:
    schema = document.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise HanFileMapError(f"Unsupported or missing schema_version: {schema!r}")
    if not isinstance(document.get("tokens"), list):
        raise HanFileMapError("Mapped document missing token list")
    if not isinstance(document.get("inventory"), list):
        raise HanFileMapError("Mapped document missing inventory list")
    source_document = document.get("source_document") or {}
    if "encoding_plan" not in source_document:
        raise HanFileMapError("Mapped document missing source_document.encoding_plan")

    inventory_len = len(document["inventory"])
    prev_char_end = 0
    prev_byte_end = None
    for expected_id, token in enumerate(document["tokens"]):
        token_id = token.get("token_id")
        if token_id != expected_id:
            raise HanFileMapError(f"Token IDs must be contiguous and ordered; expected {expected_id}, got {token_id}")
        positions = token.get("positions") or {}
        char_start = positions.get("char_start")
        char_end = positions.get("char_end")
        byte_start = positions.get("byte_start")
        byte_end = positions.get("byte_end")
        if None in {char_start, char_end, byte_start, byte_end}:
            raise HanFileMapError(f"Token {token_id} is missing position fields")
        if char_start != prev_char_end:
            raise HanFileMapError(f"Non-contiguous char spans at token {token_id}")
        if char_end < char_start:
            raise HanFileMapError(f"Invalid char span at token {token_id}")
        if prev_byte_end is not None and byte_start != prev_byte_end:
            raise HanFileMapError(f"Non-contiguous byte spans at token {token_id}")
        if byte_end < byte_start:
            raise HanFileMapError(f"Invalid byte span at token {token_id}")
        prev_char_end = char_end
        prev_byte_end = byte_end
        inv_idx = token.get("inventory_index")
        if inv_idx is not None and not (0 <= inv_idx < inventory_len):
            raise HanFileMapError(f"Token {token_id} inventory_index out of range: {inv_idx}")


def map_original_bytes(document: dict) -> Optional[bytes]:
    payload = document.get("original_payload")
    if payload is None:
        return None
    return decode_payload(payload)


def map_encoding_plan(document: dict) -> UTFEncodingPlan:
    src = document.get("source_document") or {}
    return UTFEncodingPlan.from_dict(src["encoding_plan"])


def reconstruct_original_bytes(document: dict) -> bytes:
    raw = map_original_bytes(document)
    if raw is not None:
        return raw
    plan = map_encoding_plan(document)
    text = reconstruct_text_from_tokens(
        document["tokens"],
        mode="original-text",
        component_separator=document.get("tokenization", {}).get("component_separator_default", DEFAULT_COMPONENT_SEPARATOR),
    )
    return encode_text(text, plan, preserve_bom=True)


def diff_lines(a_text: str, b_text: str, *, fromfile: str, tofile: str, context: int) -> List[str]:
    import difflib

    return list(
        difflib.unified_diff(
            a_text.splitlines(keepends=True),
            b_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
            n=context,
        )
    )


def document_text_from_original_bytes(document: dict) -> str:
    raw = reconstruct_original_bytes(document)
    plan = map_encoding_plan(document)
    return decode_text(raw, plan)


def source_slice_bytes(document: dict, token: dict) -> bytes:
    raw = reconstruct_original_bytes(document)
    pos = token["positions"]
    return raw[pos["byte_start"] : pos["byte_end"]]


__all__ = [
    "SCHEMA_VERSION",
    "MAIN_BLOCK_NAME",
    "DEFAULT_COMPONENT_SEPARATOR",
    "DEFAULT_TOKEN_SEPARATOR",
    "DEFAULT_PAYLOAD_CODEC",
    "HanFileMapError",
    "UTFEncodingPlan",
    "stderr",
    "ensure_parent_dir",
    "b64encode_bytes",
    "b64decode_bytes",
    "sha256_bytes",
    "sha256_text_utf8",
    "sha256_file",
    "iso_utc_now",
    "normalize_encoding_label",
    "is_main_block_char",
    "cp_to_uplus",
    "char_codepoints",
    "detect_utf_plan",
    "decode_text",
    "encode_text",
    "encode_payload",
    "decode_payload",
    "count_newline_kinds",
    "build_line_start_offsets",
    "offset_to_line_col",
    "classify_char",
    "unicode_name_safe",
    "deep_copy_jsonable",
    "iter_decomposition_records",
    "load_decomposition_subset",
    "build_decomposition_summary",
    "make_inventory",
    "dedupe_preserve_order",
    "build_token_records",
    "reconstruct_token_text",
    "reconstruct_text_from_tokens",
    "first_difference",
    "changed_tokens_for_mode",
    "current_text_edits",
    "write_map_document",
    "load_map_document",
    "validate_map_document",
    "map_original_bytes",
    "map_encoding_plan",
    "reconstruct_original_bytes",
    "diff_lines",
    "document_text_from_original_bytes",
    "source_slice_bytes",
]

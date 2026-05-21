#!/usr/bin/env python3
"""
Shared helpers for a readable symbolic compact serialization of the reversible
Han decomposition file map.

This format is designed to stay text-friendly while remaining exactly
reversible back to the original UTF input:

  * Han tokens (and optionally other tokens) are replaced with short visible
    symbol IDs in the body stream.
  * A readable legend maps each symbol ID back to the original token text and
    its serialized/transformed text.
  * Non-symbolized literal text is preserved directly in the body whenever that
    remains reversible; control characters and the sentinel itself are emitted
    through visible escape sequences.

The on-disk container is line-oriented and easy to diff:

    HAN-READABLE-SYMBOLIC/v1
    meta\t{...json...}
    stats\t{...json...}
    legend_begin
    symbol\t{...json...}
    ...
    legend_end
    body_begin
    |...wrapped body chunk...
    |...wrapped body chunk...

All lines after ``body_begin`` belong to the wrapped body payload. The leading
``|`` on each body line is a framing character and is stripped during load.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from han_file_decomp_map_common import (
    DEFAULT_COMPONENT_SEPARATOR,
    HanFileMapError,
    UTFEncodingPlan,
    cp_to_uplus,
    encode_text,
    iso_utc_now,
    reconstruct_token_text,
    sha256_bytes,
    sha256_text_utf8,
)

READABLE_SYMBOLIC_SCHEMA_VERSION = "han-readable-symbolic/v1"
READABLE_SYMBOLIC_MAGIC = "HAN-READABLE-SYMBOLIC/v1"
DEFAULT_SYMBOLIC_WRAP_WIDTH = 120
DEFAULT_SYMBOLIC_SENTINEL = "¤"
DEFAULT_SYMBOL_ALPHABET_PROFILE = "default"
VALID_SERIALIZATION_MODES = {"original-text", "current-text", "decomp-raw", "decomp-normalized"}

# Ordered, curated ranges with lots of visible single-codepoint symbols.
# The resulting default alphabet contains well over 1,000 code points, which
# means width=1 works for many files and width=2 covers millions of symbols.
_DEFAULT_SYMBOL_RANGES: Sequence[Tuple[int, int]] = (
    (0x2190, 0x21FF),  # Arrows
    (0x2300, 0x23FF),  # Misc Technical
    (0x2460, 0x24FF),  # Enclosed Alphanumerics / numbers
    (0x2500, 0x257F),  # Box Drawing
    (0x2580, 0x259F),  # Block Elements
    (0x25A0, 0x25FF),  # Geometric Shapes
    (0x2600, 0x26FF),  # Misc Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x27F0, 0x27FF),  # Supplemental Arrows-A
    (0x2900, 0x297F),  # Supplemental Arrows-B
    (0x2B00, 0x2BFF),  # Misc Symbols and Arrows
)


class HanReadableSymbolicError(HanFileMapError):
    """Readable symbolic serialization specific error."""


def _iter_default_symbol_alphabet() -> Iterator[str]:
    seen = set()
    for start, end in _DEFAULT_SYMBOL_RANGES:
        for cp in range(start, end + 1):
            ch = chr(cp)
            if ch in seen:
                continue
            seen.add(ch)
            cat = unicodedata.category(ch)
            if cat.startswith("C") or cat.startswith("M"):
                continue
            if ch.isspace():
                continue
            yield ch


def get_symbol_alphabet(profile: str = DEFAULT_SYMBOL_ALPHABET_PROFILE) -> List[str]:
    if profile != DEFAULT_SYMBOL_ALPHABET_PROFILE:
        raise HanReadableSymbolicError(f"Unsupported symbol alphabet profile: {profile}")
    return list(_iter_default_symbol_alphabet())


def _symbol_entry_from_token(
    token: dict,
    *,
    serialized_text: str,
    include_han_metadata: bool,
    source_inventory: Sequence[dict],
) -> dict:
    inv_idx = token.get("inventory_index")
    entry = {
        "kind": token.get("kind"),
        "original_text": token.get("text_original", ""),
        "serialized_text": serialized_text,
        "source_inventory_index": inv_idx,
    }

    if inv_idx is not None and 0 <= inv_idx < len(source_inventory):
        inv = source_inventory[inv_idx]
        entry["source_char"] = inv.get("char")
        entry["source_record_available"] = bool(inv.get("record_available"))
    else:
        entry["source_char"] = None
        entry["source_record_available"] = False

    if include_han_metadata and token.get("kind") == "han_main_block":
        decomp = token.get("decomposition") or {}
        entry["han"] = {
            "primary_raw": decomp.get("primary_raw"),
            "render_raw_default": decomp.get("render_raw_default"),
            "render_normalized_default": decomp.get("render_normalized_default"),
            "kRSUnicode": decomp.get("kRSUnicode"),
            "kTotalStrokes": decomp.get("kTotalStrokes"),
            "kDefinition": decomp.get("kDefinition"),
        }
    return entry


def _symbol_key_from_entry(entry: dict) -> Tuple[object, ...]:
    return (
        entry.get("kind"),
        entry.get("original_text"),
        entry.get("serialized_text"),
        entry.get("source_inventory_index"),
        json.dumps(entry.get("han"), ensure_ascii=False, sort_keys=True) if "han" in entry else None,
    )


def _required_symbol_id_width(symbol_count: int, alphabet_size: int) -> int:
    if alphabet_size <= 1:
        raise HanReadableSymbolicError("Symbol alphabet must contain at least two symbols")
    width = 1
    capacity = alphabet_size
    while symbol_count > capacity:
        width += 1
        capacity *= alphabet_size
    return width


def _int_to_symbol_id(index: int, alphabet: Sequence[str], width: int) -> str:
    if index < 0:
        raise HanReadableSymbolicError(f"Negative symbol index is invalid: {index}")
    base = len(alphabet)
    max_supported = base ** width
    if index >= max_supported:
        raise HanReadableSymbolicError(
            f"symbol_id width={width} cannot encode index={index}; capacity is {max_supported}"
        )
    digits = [alphabet[0]] * width
    value = index
    pos = width - 1
    while pos >= 0:
        digits[pos] = alphabet[value % base]
        value //= base
        pos -= 1
    return "".join(digits)


def _iter_escape_text(text: str, *, sentinel: str) -> Iterator[str]:
    for ch in text:
        cp = ord(ch)
        if ch == sentinel:
            yield sentinel + "~s"
        elif ch == "\n":
            yield sentinel + "~n"
        elif ch == "\r":
            yield sentinel + "~r"
        elif ch == "\t":
            yield sentinel + "~t"
        elif cp < 0x20 or cp == 0x7F or ch in {"\u2028", "\u2029"}:
            width = 4 if cp <= 0xFFFF else 6
            yield f"{sentinel}~x{cp:0{width}X};"
        else:
            yield ch


def escape_body_literal(text: str, *, sentinel: str) -> str:
    return "".join(_iter_escape_text(text, sentinel=sentinel))


def _parse_escape(body: str, start: int, *, sentinel: str) -> Tuple[str, int]:
    # body[start] == sentinel and body[start+1] == "~"
    if start + 2 >= len(body):
        raise HanReadableSymbolicError("Truncated symbolic escape at end of body")
    code = body[start + 2]
    if code == "s":
        return sentinel, start + 3
    if code == "n":
        return "\n", start + 3
    if code == "r":
        return "\r", start + 3
    if code == "t":
        return "\t", start + 3
    if code == "x":
        end = body.find(";", start + 3)
        if end == -1:
            raise HanReadableSymbolicError("Unterminated hex escape in symbolic body")
        hex_part = body[start + 3 : end]
        if not hex_part or any(ch not in "0123456789ABCDEFabcdef" for ch in hex_part):
            raise HanReadableSymbolicError(f"Invalid hex escape in symbolic body: {hex_part!r}")
        cp = int(hex_part, 16)
        try:
            return chr(cp), end + 1
        except ValueError as exc:
            raise HanReadableSymbolicError(f"Invalid code point in symbolic body escape: U+{cp:04X}") from exc
    raise HanReadableSymbolicError(f"Unknown symbolic escape code: ~{code}")


def _should_symbolize_token(
    token: dict,
    *,
    serialized_text: str,
    symbolize_non_han: bool,
) -> bool:
    if token.get("kind") == "han_main_block":
        return True
    if symbolize_non_han:
        return True
    # If the transformed/current rendering would differ from the original token,
    # keep the token in the legend so the container stays reversible.
    return serialized_text != token.get("text_original", "")


def build_readable_symbolic_document_from_map(
    mapped_document: dict,
    *,
    mode: str = "decomp-normalized",
    component_separator: str = DEFAULT_COMPONENT_SEPARATOR,
    preserve_non_han: bool = True,
    missing_han_fallback: str = "current",
    include_han_metadata: bool = False,
    symbolize_non_han: bool = False,
    sentinel: str = DEFAULT_SYMBOLIC_SENTINEL,
    symbol_alphabet_profile: str = DEFAULT_SYMBOL_ALPHABET_PROFILE,
    symbol_id_width: Optional[int] = None,
) -> dict:
    if mode not in VALID_SERIALIZATION_MODES:
        raise HanReadableSymbolicError(f"Unsupported serialization mode: {mode}")
    if len(sentinel) != 1:
        raise HanReadableSymbolicError("Sentinel must be exactly one Unicode code point")

    source_document = mapped_document.get("source_document") or {}
    tokens = mapped_document.get("tokens") or []
    inventory = mapped_document.get("inventory") or []

    alphabet = get_symbol_alphabet(symbol_alphabet_profile)
    if sentinel in alphabet:
        alphabet = [ch for ch in alphabet if ch != sentinel]
    if not alphabet:
        raise HanReadableSymbolicError("Resolved symbol alphabet is empty after removing the sentinel")

    symbol_entries: List[dict] = []
    symbol_index: Dict[Tuple[object, ...], int] = {}
    symbol_occurrence_counts: Dict[int, int] = {}
    token_symbol_refs: List[Optional[int]] = []
    token_serialized_texts: List[str] = []

    for token in tokens:
        serialized_text = reconstruct_token_text(
            token,
            mode=mode,
            component_separator=component_separator,
            preserve_non_han=preserve_non_han,
            missing_han_fallback=missing_han_fallback,
        )
        token_serialized_texts.append(serialized_text)
        should_symbolize = _should_symbolize_token(
            token,
            serialized_text=serialized_text,
            symbolize_non_han=symbolize_non_han,
        )
        if not should_symbolize:
            token_symbol_refs.append(None)
            continue

        entry = _symbol_entry_from_token(
            token,
            serialized_text=serialized_text,
            include_han_metadata=include_han_metadata,
            source_inventory=inventory,
        )
        key = _symbol_key_from_entry(entry)
        sid = symbol_index.get(key)
        if sid is None:
            sid = len(symbol_entries)
            symbol_index[key] = sid
            symbol_entries.append(entry)
        symbol_occurrence_counts[sid] = symbol_occurrence_counts.get(sid, 0) + 1
        token_symbol_refs.append(sid)

    if symbol_id_width is None:
        symbol_id_width = max(1, _required_symbol_id_width(len(symbol_entries) or 1, len(alphabet)))
    else:
        symbol_id_width = int(symbol_id_width)
        if symbol_id_width <= 0:
            raise HanReadableSymbolicError("symbol_id_width must be a positive integer")
        max_capacity = len(alphabet) ** symbol_id_width
        if len(symbol_entries) > max_capacity:
            raise HanReadableSymbolicError(
                f"symbol_id_width={symbol_id_width} cannot encode {len(symbol_entries)} symbols; "
                f"capacity is {max_capacity}"
            )

    symbol_lookup_by_index: Dict[int, dict] = {}
    for sid, entry in enumerate(symbol_entries):
        symbol_id = _int_to_symbol_id(sid, alphabet, symbol_id_width)
        entry["symbol_index"] = sid
        entry["symbol_id"] = symbol_id
        entry["occurrence_count"] = symbol_occurrence_counts.get(sid, 0)
        entry["original_codepoints"] = [cp_to_uplus(ord(ch)) for ch in entry["original_text"]]
        entry["serialized_codepoints"] = [cp_to_uplus(ord(ch)) for ch in entry["serialized_text"]]
        symbol_lookup_by_index[sid] = entry

    body_parts: List[str] = []
    for token, sid, serialized_text in zip(tokens, token_symbol_refs, token_serialized_texts):
        if sid is not None:
            body_parts.append(sentinel)
            body_parts.append(symbol_lookup_by_index[sid]["symbol_id"])
            continue
        # Raw literals are emitted only when they already match the serialized
        # view. They still need escaping for the body container grammar.
        original_text = token.get("text_original", "")
        if serialized_text != original_text:
            raise HanReadableSymbolicError(
                "Encountered a non-symbolized token whose serialized text does not match the original; "
                "this would break reversibility."
            )
        body_parts.append(escape_body_literal(original_text, sentinel=sentinel))
    body = "".join(body_parts)

    symbolic_document = {
        "schema_version": READABLE_SYMBOLIC_SCHEMA_VERSION,
        "generator": {
            "script": "han_file_symbolic_serialize.py",
            "version": "1.0.0",
            "created_at_utc": iso_utc_now(),
        },
        "source_document": {
            "path": source_document.get("path"),
            "filename": source_document.get("filename"),
            "byte_length": source_document.get("byte_length"),
            "sha256_bytes": source_document.get("sha256_bytes"),
            "text_length_codepoints": source_document.get("text_length_codepoints"),
            "text_sha256_utf8": source_document.get("text_sha256_utf8"),
            "line_count": source_document.get("line_count"),
            "newline_counts": source_document.get("newline_counts"),
            "encoding_plan": source_document.get("encoding_plan"),
        },
        "source_map": {
            "schema_version": mapped_document.get("schema_version"),
            "generator": mapped_document.get("generator"),
            "decomposition_dataset": mapped_document.get("decomposition_dataset"),
            "inventory_count": len(inventory),
            "token_count": len(tokens),
        },
        "serialization": {
            "mode": mode,
            "component_separator": component_separator,
            "preserve_non_han": preserve_non_han,
            "missing_han_fallback": missing_han_fallback,
            "include_han_metadata": include_han_metadata,
            "symbolize_non_han": symbolize_non_han,
            "sentinel": sentinel,
            "symbol_alphabet_profile": symbol_alphabet_profile,
            "symbol_id_width": symbol_id_width,
        },
        "stats": {
            "token_count": len(tokens),
            "symbol_entry_count": len(symbol_entries),
            "symbolized_token_count": sum(1 for sid in token_symbol_refs if sid is not None),
            "han_symbol_entry_count": sum(1 for entry in symbol_entries if entry.get("kind") == "han_main_block"),
            "han_symbolized_token_count": sum(
                entry.get("occurrence_count", 0)
                for entry in symbol_entries
                if entry.get("kind") == "han_main_block"
            ),
            "body_length_codepoints": len(body),
            "body_sha256_utf8": sha256_text_utf8(body),
            "original_text_length_codepoints": sum(len(token.get("text_original", "")) for token in tokens),
            "symbol_id_width": symbol_id_width,
        },
        "symbol_table": symbol_entries,
        "body": body,
    }
    validate_readable_symbolic_document(symbolic_document)
    return symbolic_document


def symbolic_encoding_plan(document: dict) -> UTFEncodingPlan:
    src = document.get("source_document") or {}
    plan = src.get("encoding_plan")
    if not isinstance(plan, dict):
        raise HanReadableSymbolicError("Symbolic document missing source_document.encoding_plan")
    return UTFEncodingPlan.from_dict(plan)


def _symbol_lookup(document: dict) -> Dict[str, dict]:
    return {entry["symbol_id"]: entry for entry in document.get("symbol_table", [])}


def iter_decoded_body_units(document: dict) -> Iterator[Tuple[str, str]]:
    """
    Yield pairs ``(kind, value)`` where kind is either ``literal`` or ``symbol``.

    For ``literal`` units, value is the literal character to append.
    For ``symbol`` units, value is the symbol_id string.
    """
    validate_readable_symbolic_document(document)
    serialization = document.get("serialization") or {}
    sentinel = serialization["sentinel"]
    id_width = serialization["symbol_id_width"]
    body = document.get("body", "")
    symbol_lookup = _symbol_lookup(document)

    i = 0
    while i < len(body):
        ch = body[i]
        if ch != sentinel:
            yield "literal", ch
            i += 1
            continue

        if i + 1 >= len(body):
            raise HanReadableSymbolicError("Trailing sentinel at end of symbolic body")
        if body[i + 1] == "~":
            literal, i = _parse_escape(body, i, sentinel=sentinel)
            yield "literal", literal
            continue

        end = i + 1 + id_width
        symbol_id = body[i + 1 : end]
        if len(symbol_id) != id_width:
            raise HanReadableSymbolicError("Truncated symbol ID at end of symbolic body")
        if symbol_id not in symbol_lookup:
            raise HanReadableSymbolicError(f"Unknown symbol ID in symbolic body: {symbol_id!r}")
        yield "symbol", symbol_id
        i = end


def reconstruct_original_text(document: dict) -> str:
    symbol_lookup = _symbol_lookup(document)
    parts: List[str] = []
    for kind, value in iter_decoded_body_units(document):
        if kind == "literal":
            parts.append(value)
        else:
            parts.append(symbol_lookup[value].get("original_text", ""))
    return "".join(parts)


def reconstruct_serialized_text(document: dict) -> str:
    symbol_lookup = _symbol_lookup(document)
    parts: List[str] = []
    for kind, value in iter_decoded_body_units(document):
        if kind == "literal":
            parts.append(value)
        else:
            parts.append(symbol_lookup[value].get("serialized_text", ""))
    return "".join(parts)


def reconstruct_original_bytes(document: dict, *, errors: Optional[str] = None) -> bytes:
    plan = symbolic_encoding_plan(document)
    text = reconstruct_original_text(document)
    return encode_text(text, plan, preserve_bom=True, errors=errors)


def readable_symbolic_compare_original_hash(document: dict, *, errors: Optional[str] = None) -> dict:
    recovered = reconstruct_original_bytes(document, errors=errors)
    recovered_hash = sha256_bytes(recovered)
    source_hash = (document.get("source_document") or {}).get("sha256_bytes")
    return {
        "bytes_equal_to_stored_hash": (source_hash == recovered_hash) if source_hash else None,
        "recovered_sha256_bytes": recovered_hash,
        "stored_sha256_bytes": source_hash,
        "recovered_byte_length": len(recovered),
        "stored_byte_length": (document.get("source_document") or {}).get("byte_length"),
    }


def validate_readable_symbolic_document(document: dict) -> None:
    schema = document.get("schema_version")
    if schema != READABLE_SYMBOLIC_SCHEMA_VERSION:
        raise HanReadableSymbolicError(f"Unsupported or missing schema_version: {schema!r}")

    serialization = document.get("serialization")
    if not isinstance(serialization, dict):
        raise HanReadableSymbolicError("Symbolic document missing serialization block")
    if serialization.get("mode") not in VALID_SERIALIZATION_MODES:
        raise HanReadableSymbolicError("Symbolic document has unsupported serialization mode")
    sentinel = serialization.get("sentinel")
    if not isinstance(sentinel, str) or len(sentinel) != 1:
        raise HanReadableSymbolicError("Symbolic document sentinel must be a single code point")
    id_width = serialization.get("symbol_id_width")
    if not isinstance(id_width, int) or id_width <= 0:
        raise HanReadableSymbolicError("Symbolic document missing a valid symbol_id_width")

    symbol_table = document.get("symbol_table")
    if not isinstance(symbol_table, list):
        raise HanReadableSymbolicError("Symbolic document missing symbol_table list")
    seen_ids = set()
    for idx, entry in enumerate(symbol_table):
        symbol_id = entry.get("symbol_id")
        if not isinstance(symbol_id, str) or len(symbol_id) != id_width:
            raise HanReadableSymbolicError(
                f"symbol_table entry {idx} has invalid symbol_id width: {symbol_id!r}"
            )
        if symbol_id in seen_ids:
            raise HanReadableSymbolicError(f"Duplicate symbol_id in symbol_table: {symbol_id!r}")
        seen_ids.add(symbol_id)
        if "original_text" not in entry or "serialized_text" not in entry:
            raise HanReadableSymbolicError(f"symbol_table entry {idx} missing required text fields")

    if not isinstance(document.get("body"), str):
        raise HanReadableSymbolicError("Symbolic document missing body string")

    source_document = document.get("source_document") or {}
    if "encoding_plan" not in source_document:
        raise HanReadableSymbolicError("Symbolic document missing source_document.encoding_plan")

    # Parse once to verify the body is structurally valid and to check token count.
    token_count = 0
    for _kind, _value in iter_decoded_body_units_no_validate(document, symbol_table=symbol_table):
        token_count += 1
    expected = (document.get("stats") or {}).get("token_count")
    if expected is not None and expected != token_count:
        raise HanReadableSymbolicError(
            f"Symbolic body token count mismatch: expected {expected}, decoded {token_count}"
        )


def iter_decoded_body_units_no_validate(
    document: dict,
    *,
    symbol_table: Optional[Sequence[dict]] = None,
) -> Iterator[Tuple[str, str]]:
    serialization = document.get("serialization") or {}
    sentinel = serialization["sentinel"]
    id_width = serialization["symbol_id_width"]
    body = document.get("body", "")
    if symbol_table is None:
        symbol_table = document.get("symbol_table", [])
    symbol_lookup = {entry["symbol_id"]: entry for entry in symbol_table}

    i = 0
    while i < len(body):
        ch = body[i]
        if ch != sentinel:
            yield "literal", ch
            i += 1
            continue
        if i + 1 >= len(body):
            raise HanReadableSymbolicError("Trailing sentinel at end of symbolic body")
        if body[i + 1] == "~":
            literal, i = _parse_escape(body, i, sentinel=sentinel)
            yield "literal", literal
            continue
        end = i + 1 + id_width
        symbol_id = body[i + 1 : end]
        if len(symbol_id) != id_width:
            raise HanReadableSymbolicError("Truncated symbol ID at end of symbolic body")
        if symbol_id not in symbol_lookup:
            raise HanReadableSymbolicError(f"Unknown symbol ID in symbolic body: {symbol_id!r}")
        yield "symbol", symbol_id
        i = end


def _wrap_text(text: str, width: int) -> List[str]:
    if width <= 0:
        return [text] if text else []
    return [text[i : i + width] for i in range(0, len(text), width)]


def write_readable_symbolic_document(
    document: dict,
    output_path: Path,
    *,
    wrap_width: int = DEFAULT_SYMBOLIC_WRAP_WIDTH,
) -> None:
    validate_readable_symbolic_document(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": document.get("schema_version"),
        "generator": document.get("generator"),
        "source_document": document.get("source_document"),
        "source_map": document.get("source_map"),
        "serialization": document.get("serialization"),
    }
    stats = document.get("stats") or {}

    with output_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(READABLE_SYMBOLIC_MAGIC + "\n")
        fh.write("meta\t")
        fh.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")
        fh.write("stats\t")
        fh.write(json.dumps(stats, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")
        fh.write("legend_begin\n")
        for entry in document.get("symbol_table", []):
            fh.write("symbol\t")
            fh.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
        fh.write("legend_end\n")
        fh.write("body_begin\n")
        body = document.get("body", "")
        for chunk in _wrap_text(body, wrap_width):
            fh.write("|")
            fh.write(chunk)
            fh.write("\n")
        if body == "":
            # Represent the empty body explicitly for readability and stable load.
            fh.write("|\n")


def load_readable_symbolic_document(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh]
    if not lines or lines[0] != READABLE_SYMBOLIC_MAGIC:
        raise HanReadableSymbolicError(f"Missing or invalid symbolic container magic in {path}")

    meta = None
    stats = None
    symbol_table: List[dict] = []
    body_chunks: List[str] = []
    i = 1
    in_legend = False
    in_body = False
    while i < len(lines):
        line = lines[i]
        i += 1
        if in_body:
            if not line.startswith("|"):
                raise HanReadableSymbolicError(
                    f"Malformed body chunk line in {path}; expected leading '|': {line!r}"
                )
            body_chunks.append(line[1:])
            continue
        if in_legend:
            if line == "legend_end":
                in_legend = False
                continue
            if not line.startswith("symbol\t"):
                raise HanReadableSymbolicError(f"Malformed legend line in {path}: {line!r}")
            try:
                entry = json.loads(line.split("\t", 1)[1])
            except Exception as exc:
                raise HanReadableSymbolicError(f"Failed to parse legend entry JSON: {exc}") from exc
            symbol_table.append(entry)
            continue

        if line.startswith("meta\t"):
            try:
                meta = json.loads(line.split("\t", 1)[1])
            except Exception as exc:
                raise HanReadableSymbolicError(f"Failed to parse meta JSON: {exc}") from exc
            continue
        if line.startswith("stats\t"):
            try:
                stats = json.loads(line.split("\t", 1)[1])
            except Exception as exc:
                raise HanReadableSymbolicError(f"Failed to parse stats JSON: {exc}") from exc
            continue
        if line == "legend_begin":
            in_legend = True
            continue
        if line == "body_begin":
            in_body = True
            continue
        if line == "":
            continue
        raise HanReadableSymbolicError(f"Unexpected line in symbolic container: {line!r}")

    if meta is None:
        raise HanReadableSymbolicError(f"Symbolic container missing meta block: {path}")
    if stats is None:
        raise HanReadableSymbolicError(f"Symbolic container missing stats block: {path}")
    if not in_body:
        raise HanReadableSymbolicError(f"Symbolic container missing body_begin: {path}")

    document = {
        "schema_version": meta.get("schema_version"),
        "generator": meta.get("generator"),
        "source_document": meta.get("source_document"),
        "source_map": meta.get("source_map"),
        "serialization": meta.get("serialization"),
        "stats": stats,
        "symbol_table": symbol_table,
        "body": "".join(body_chunks) if body_chunks != [""] else "",
    }
    validate_readable_symbolic_document(document)
    return document


__all__ = [
    "READABLE_SYMBOLIC_SCHEMA_VERSION",
    "READABLE_SYMBOLIC_MAGIC",
    "DEFAULT_SYMBOLIC_WRAP_WIDTH",
    "DEFAULT_SYMBOLIC_SENTINEL",
    "DEFAULT_SYMBOL_ALPHABET_PROFILE",
    "VALID_SERIALIZATION_MODES",
    "HanReadableSymbolicError",
    "get_symbol_alphabet",
    "escape_body_literal",
    "build_readable_symbolic_document_from_map",
    "symbolic_encoding_plan",
    "validate_readable_symbolic_document",
    "iter_decoded_body_units",
    "reconstruct_original_text",
    "reconstruct_serialized_text",
    "reconstruct_original_bytes",
    "readable_symbolic_compare_original_hash",
    "write_readable_symbolic_document",
    "load_readable_symbolic_document",
]

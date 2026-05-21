
#!/usr/bin/env python3
"""
Build recursive subcomponent decompositions for Chinese Han characters in the
main Unicode block (U+4E00..U+9FFF).

The script combines:
  * official Unicode/UCD/Unihan metadata (UnicodeData.txt, Unihan.zip,
    CJKRadicals.txt, EquivalentUnifiedIdeograph.txt)
  * a decomposition source such as cjkvi-ids, CHISE, or cjk-decomp

Goals
-----
1. keep the official radical/stroke metadata separate from structural
   decomposition;
2. preserve exact source decompositions;
3. derive a second "normalized radicals" layer for downstream search/indexing.

The implementation is intentionally strict about source formats, but it also
contains enough heuristics to cope with the real-world quirks in CHISE/cjkvi
IDS data:
  * entity references such as &U-i003+51AC;
  * angle-bracket tokens like <CJK RADICAL MEAT>;
  * optional CHISE @apparent= metadata;
  * alternative IDS strings on the same line;
  * current Unicode IDS grammar, including unary IDCs and U+31EF.

The script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import re
import shutil
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MAIN_BLOCK_START = 0x4E00
MAIN_BLOCK_END = 0x9FFF
MAIN_BLOCK_NAME = "CJK Unified Ideographs"

LATEST_UCD_BASE = "https://www.unicode.org/Public/UCD/latest/ucd/"

DEFAULT_URLS = {
    "unicode_data": f"{LATEST_UCD_BASE}UnicodeData.txt",
    "unihan_zip": f"{LATEST_UCD_BASE}Unihan.zip",
    "cjk_radicals": f"{LATEST_UCD_BASE}CJKRadicals.txt",
    "equivalent_unified": f"{LATEST_UCD_BASE}EquivalentUnifiedIdeograph.txt",
    "cjkvi": "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt",
    "chise": "https://raw.githubusercontent.com/osfans/chise-ids/master/IDS-UCS-Basic.txt",
    "cjk_decomp": "https://raw.githubusercontent.com/amake/cjk-decomp/master/cjk-decomp.txt",
}

DOWNLOAD_FILENAMES = {
    "unicode_data": "UnicodeData.txt",
    "unihan_zip": "Unihan.zip",
    "cjk_radicals": "CJKRadicals.txt",
    "equivalent_unified": "EquivalentUnifiedIdeograph.txt",
    "cjkvi": "ids.txt",
    "chise": "IDS-UCS-Basic.txt",
    "cjk_decomp": "cjk-decomp.txt",
}

IDS_UNARY = {"⿾", "⿿"}
IDS_BINARY = {"⿰", "⿱", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻", "⿼", "⿽", "㇯"}
IDS_TRINARY = {"⿲", "⿳"}
IDS_ARITY = {op: 1 for op in IDS_UNARY}
IDS_ARITY.update({op: 2 for op in IDS_BINARY})
IDS_ARITY.update({op: 3 for op in IDS_TRINARY})

# Variant normalization is intentionally layered:
# - conservative: obvious radical/component allographs and simplification forms
# - aggressive: adds a few common component-to-radical reductions
CONSERVATIVE_VARIANT_MAP = {
    "亻": "人",
    "⺅": "人",
    "氵": "水",
    "氺": "水",
    "忄": "心",
    "⺖": "心",
    "㣺": "心",
    "⺗": "心",
    "扌": "手",
    "龵": "手",
    "犭": "犬",
    "礻": "示",
    "⺬": "示",
    "衤": "衣",
    "讠": "言",
    "訁": "言",
    "钅": "金",
    "釒": "金",
    "饣": "食",
    "飠": "食",
    "纟": "糸",
    "糹": "糸",
    "辶": "辵",
    "⻌": "辵",
    "⻍": "辵",
    "攵": "攴",
    "爫": "爪",
    "灬": "火",
    "⺼": "肉",
    "艹": "艸",
    "⺾": "艸",
    "⺿": "艸",
    "罒": "网",
    "⺲": "网",
    "罓": "网",
    "覀": "西",
}
AGGRESSIVE_EXTRA_VARIANT_MAP = {
    "刂": "刀",
    "⺉": "刀",
    "⺈": "刀",
    "⺌": "小",
    "⺍": "小",
    "⺨": "犬",
    "⺡": "水",
    "⺘": "手",
    "⺩": "玉",
    "⻎": "辵",
    "⻏": "邑",
    "⻖": "阜",
    "⻂": "衣",
    "⻑": "長",
    "⻒": "長",
    "⻨": "麥",
    "⻩": "黃",
    "⻪": "黽",
}
UNRESOLVED_PREFIXES = ("&", "<", "U+", "U-")
TRAILING_PAREN_COMMENT_RE = re.compile(r"\s+\([^)]+\)\s*$")
INLINE_HASH_COMMENT_RE = re.compile(r"\s+#.*$")
RS_UNICODE_RE = re.compile(r"^(?P<num>\d{1,3})(?P<marks>'{0,3})\.(?P<residual>-?\d{1,3})$")
U_TOKEN_RE = re.compile(r"^(?:U\+([0-9A-Fa-f]{4,6})|U-([0-9A-Fa-f]{8}))$")
U_TOKEN_ANYWHERE_RE = re.compile(r"(?:U\+([0-9A-Fa-f]{4,6})|U-([0-9A-Fa-f]{8}))")
CHISE_LIKE_LEAD_RE = re.compile(
    r"^(?P<cp>U\+[0-9A-Fa-f]{4,6}|U-[0-9A-Fa-f]{8})"
    r"(?:\t+|\s+)"
    r"(?P<char>\S)"
    r"(?:\t+|\s+)"
    r"(?P<rest>.+?)\s*$"
)
CJK_DECOMP_RE = re.compile(r"^(?P<target>[^:#\s][^:]*)\:(?P<kind>[^()]+)\((?P<parts>.*)\)\s*$")


class HanDecompError(Exception):
    """Base exception for this script."""


class ParseError(HanDecompError):
    """Raised when a decomposition cannot be parsed."""


@dataclasses.dataclass(frozen=True)
class IDSNode:
    kind: str  # "leaf" | "op"
    value: str
    children: Tuple["IDSNode", ...] = dataclasses.field(default_factory=tuple)
    scheme: str = "ids"

    @property
    def is_leaf(self) -> bool:
        return self.kind == "leaf"


@dataclasses.dataclass
class RSUnicodeValue:
    raw: str
    radical_key: str
    radical_number: int
    simplification_marks: int
    residual_strokes: int
    radical_char: Optional[str]
    radical_unified_char: Optional[str]

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "radical_key": self.radical_key,
            "radical_number": self.radical_number,
            "simplification_marks": self.simplification_marks,
            "residual_strokes": self.residual_strokes,
            "radical_char": self.radical_char,
            "radical_unified_char": self.radical_unified_char,
        }


@dataclasses.dataclass
class DecompositionEntry:
    key: str
    char: Optional[str]
    source_format: str
    primary_raw: Optional[str] = None
    primary_tree: Optional[IDSNode] = None
    alternative_raw: List[str] = dataclasses.field(default_factory=list)
    apparent_raw: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, List[str]] = dataclasses.field(default_factory=lambda: defaultdict(list))
    source_lines: List[int] = dataclasses.field(default_factory=list)

    def merge_from(self, other: "DecompositionEntry") -> None:
        if self.key != other.key:
            raise ValueError("Cannot merge entries for different keys")
        if self.char is None and other.char is not None:
            self.char = other.char
        if self.primary_raw is None and other.primary_raw is not None:
            self.primary_raw = other.primary_raw
            self.primary_tree = other.primary_tree
        elif other.primary_raw and other.primary_raw != self.primary_raw:
            if other.primary_raw not in self.alternative_raw:
                self.alternative_raw.append(other.primary_raw)
        for item in other.alternative_raw:
            if item not in self.alternative_raw and item != self.primary_raw:
                self.alternative_raw.append(item)
        for item in other.apparent_raw:
            if item not in self.apparent_raw:
                self.apparent_raw.append(item)
        for key, values in other.metadata.items():
            existing = self.metadata.setdefault(key, [])
            for value in values:
                if value not in existing:
                    existing.append(value)
        for line in other.source_lines:
            if line not in self.source_lines:
                self.source_lines.append(line)


@dataclasses.dataclass(frozen=True)
class LeafContext:
    parent_op: Optional[str]
    scheme: Optional[str]
    child_index: Optional[int]
    sibling_count: Optional[int]


@dataclasses.dataclass
class BuildContext:
    assigned_main_block: Dict[int, str]
    name_lookup: Dict[str, str]
    radical_map: Dict[str, Tuple[Optional[str], Optional[str]]]
    equivalent_map: Dict[str, str]
    unihan: Dict[int, Dict[str, str]]
    decomp_db: Dict[str, DecompositionEntry]
    entity_map: Dict[str, str]
    normalization: str
    max_depth: int


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_ucd_name(name: str) -> str:
    return re.sub(r"[\s\-_]+", " ", name.strip().upper())


def parse_codepoint_token(token: str) -> Optional[int]:
    m = U_TOKEN_RE.match(token)
    if not m:
        return None
    hex_text = m.group(1) or m.group(2)
    if hex_text is None:
        return None
    return int(hex_text, 16)


def char_from_codepoint_token(token: str) -> Optional[str]:
    cp = parse_codepoint_token(token)
    if cp is None or cp > 0x10FFFF:
        return None
    return chr(cp)


def is_main_block(cp: int) -> bool:
    return MAIN_BLOCK_START <= cp <= MAIN_BLOCK_END


def cp_to_uplus(cp: int) -> str:
    if cp <= 0xFFFF:
        return f"U+{cp:04X}"
    return f"U+{cp:06X}"


def serialize_node(node: IDSNode) -> dict:
    if node.is_leaf:
        return {"kind": "leaf", "value": node.value, "scheme": node.scheme}
    return {
        "kind": "op",
        "value": node.value,
        "scheme": node.scheme,
        "children": [serialize_node(child) for child in node.children],
    }


def node_surface(node: IDSNode) -> str:
    if node.is_leaf:
        return node.value
    if node.scheme == "ids":
        return node.value + "".join(node_surface(child) for child in node.children)
    return f"{node.value}({','.join(node_surface(child) for child in node.children)})"


def immediate_component_surfaces(node: Optional[IDSNode]) -> List[str]:
    if node is None or node.is_leaf:
        return []
    return [node_surface(child) for child in node.children]


def collect_leaf_tokens(node: IDSNode, context: Optional[LeafContext] = None) -> List[Tuple[str, LeafContext]]:
    if node.is_leaf:
        if context is None:
            context = LeafContext(None, None, None, None)
        return [(node.value, context)]
    out: List[Tuple[str, LeafContext]] = []
    n = len(node.children)
    for idx, child in enumerate(node.children):
        child_context = LeafContext(node.value, node.scheme, idx, n)
        out.extend(collect_leaf_tokens(child, child_context))
    return out


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def is_unresolved_token(token: str) -> bool:
    if not token:
        return True
    if token.startswith(UNRESOLVED_PREFIXES):
        return True
    if token.isdigit():
        return True
    # A token can be more than one code point if it is still unresolved or if it
    # is a CHISE/cjk-decomp synthetic identifier. Surface Unicode leaves should
    # generally collapse to a single Unicode scalar here.
    return len(token) != 1


def load_json_mapping(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise HanDecompError(f"Entity map must be a JSON object: {path}")
    out = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise HanDecompError("Entity map keys and values must be strings")
        out[key] = value
    return out


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        return
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    stderr(f"Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    except urllib.error.URLError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise HanDecompError(f"Failed to download {url}: {exc}") from exc
    tmp.replace(dest)


def download_required_files(
    data_dir: Path,
    decomp_source: str,
    force: bool = False,
    decomp_url: Optional[str] = None,
) -> Dict[str, Path]:
    downloads_dir = data_dir / "downloads"
    ensure_dir(downloads_dir)
    paths = {
        "unicode_data": downloads_dir / DOWNLOAD_FILENAMES["unicode_data"],
        "unihan_zip": downloads_dir / DOWNLOAD_FILENAMES["unihan_zip"],
        "cjk_radicals": downloads_dir / DOWNLOAD_FILENAMES["cjk_radicals"],
        "equivalent_unified": downloads_dir / DOWNLOAD_FILENAMES["equivalent_unified"],
    }
    for key in ("unicode_data", "unihan_zip", "cjk_radicals", "equivalent_unified"):
        download_file(DEFAULT_URLS[key], paths[key], force=force)

    if decomp_source != "none":
        decomp_key = "cjkvi" if decomp_source == "auto" else decomp_source
        if decomp_url is None:
            if decomp_key not in DEFAULT_URLS:
                raise HanDecompError(f"No default URL for decomposition source: {decomp_source}")
            decomp_url = DEFAULT_URLS[decomp_key]
        filename = DOWNLOAD_FILENAMES.get(decomp_key, Path(urllib.parse.urlparse(decomp_url).path).name or "decomp.txt")
        paths["decomp"] = downloads_dir / filename
        download_file(decomp_url, paths["decomp"], force=force)
    return paths


def parse_unicode_data(path: Path) -> Tuple[Dict[int, str], Dict[str, str]]:
    """
    Returns:
      assigned_main_block: cp -> name
      name_lookup: normalized UCD character name -> char
    """
    assigned: Dict[int, str] = {}
    name_lookup: Dict[str, str] = {}
    pending_range_start: Optional[int] = None
    pending_range_name: Optional[str] = None

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            fields = line.split(";")
            if len(fields) < 2:
                continue
            cp = int(fields[0], 16)
            name = fields[1]

            if name and not name.startswith("<"):
                name_lookup[normalize_ucd_name(name)] = chr(cp)

            if name.endswith(", First>"):
                pending_range_start = cp
                pending_range_name = name
                continue

            if name.endswith(", Last>"):
                if pending_range_start is None or pending_range_name is None:
                    raise HanDecompError(f"Malformed UnicodeData range near {fields[0]}")
                start = pending_range_start
                end = cp
                overlap_start = max(start, MAIN_BLOCK_START)
                overlap_end = min(end, MAIN_BLOCK_END)
                if overlap_start <= overlap_end:
                    for current in range(overlap_start, overlap_end + 1):
                        assigned[current] = f"CJK UNIFIED IDEOGRAPH-{current:04X}"
                pending_range_start = None
                pending_range_name = None
                continue

            if is_main_block(cp):
                if name.startswith("<") and "CJK Ideograph" in name:
                    assigned[cp] = f"CJK UNIFIED IDEOGRAPH-{cp:04X}"
                else:
                    assigned[cp] = name

    return assigned, name_lookup


def parse_cjk_radicals(path: Path) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Returns radical_key -> (radical_char, unified_char)
    radical_key keeps apostrophes, e.g. "90'".
    """
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 3:
                continue
            radical_key = parts[0]
            radical_cp = int(parts[1], 16) if parts[1] else None
            unified_cp = int(parts[2], 16) if parts[2] else None
            radical_char = chr(radical_cp) if radical_cp is not None else None
            unified_char = chr(unified_cp) if unified_cp is not None else None
            out[radical_key] = (radical_char, unified_char)
    return out


def parse_equivalent_unified(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 3:
                continue
            try:
                src_cp = int(parts[0], 16)
                dst_cp = int(parts[2], 16)
            except ValueError:
                continue
            out[chr(src_cp)] = chr(dst_cp)
    return out


def parse_unihan_zip(path: Path, needed_fields: Optional[Sequence[str]] = None) -> Dict[int, Dict[str, str]]:
    if needed_fields is None:
        needed_fields = ("kRSUnicode", "kTotalStrokes", "kDefinition")
    needed = set(needed_fields)
    out: Dict[int, Dict[str, str]] = defaultdict(dict)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            with zf.open(name) as fh:
                for raw_line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if not raw_line or raw_line.startswith("#"):
                        continue
                    parts = raw_line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    cp_field, prop, value = parts
                    if prop not in needed:
                        continue
                    cp = parse_codepoint_token(cp_field)
                    if cp is None or not is_main_block(cp):
                        continue
                    out[cp][prop] = value
    return out


def parse_rs_unicode(raw_value: Optional[str], radical_map: Dict[str, Tuple[Optional[str], Optional[str]]]) -> List[RSUnicodeValue]:
    if not raw_value:
        return []
    values: List[RSUnicodeValue] = []
    for item in raw_value.split():
        m = RS_UNICODE_RE.match(item)
        if not m:
            continue
        radical_key = f"{m.group('num')}{m.group('marks')}"
        radical_number = int(m.group("num"))
        marks = len(m.group("marks"))
        residual = int(m.group("residual"))
        radical_char, radical_unified = radical_map.get(radical_key, (None, None))
        values.append(
            RSUnicodeValue(
                raw=item,
                radical_key=radical_key,
                radical_number=radical_number,
                simplification_marks=marks,
                residual_strokes=residual,
                radical_char=radical_char,
                radical_unified_char=radical_unified,
            )
        )
    return values


def parse_total_strokes(raw_value: Optional[str]) -> List[int]:
    if not raw_value:
        return []
    values: List[int] = []
    for item in raw_value.split():
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def tokenize_ids(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "&":
            j = text.find(";", i + 1)
            if j != -1:
                tokens.append(text[i : j + 1])
                i = j + 1
                continue
        if ch == "<":
            j = text.find(">", i + 1)
            if j != -1:
                tokens.append(text[i : j + 1])
                i = j + 1
                continue
        if text.startswith("U+", i) or text.startswith("U-", i):
            m = U_TOKEN_ANYWHERE_RE.match(text[i:])
            if m:
                tok = m.group(0)
                tokens.append(tok)
                i += len(tok)
                continue
        tokens.append(ch)
        i += 1
    return tokens


def parse_ids_from_tokens(tokens: Sequence[str], start: int = 0) -> Tuple[IDSNode, int]:
    if start >= len(tokens):
        raise ParseError("Unexpected end of IDS")
    tok = tokens[start]
    if tok in IDS_UNARY:
        child, next_index = parse_ids_from_tokens(tokens, start + 1)
        return IDSNode("op", tok, (child,), "ids"), next_index
    if tok in IDS_BINARY:
        left, idx = parse_ids_from_tokens(tokens, start + 1)
        right, idx = parse_ids_from_tokens(tokens, idx)
        return IDSNode("op", tok, (left, right), "ids"), idx
    if tok in IDS_TRINARY:
        first, idx = parse_ids_from_tokens(tokens, start + 1)
        second, idx = parse_ids_from_tokens(tokens, idx)
        third, idx = parse_ids_from_tokens(tokens, idx)
        return IDSNode("op", tok, (first, second, third), "ids"), idx
    return IDSNode("leaf", tok, (), "ids"), start + 1


def extract_ids_candidates(text: str) -> List[str]:
    cleaned = TRAILING_PAREN_COMMENT_RE.sub("", text.strip())
    cleaned = INLINE_HASH_COMMENT_RE.sub("", cleaned).strip()
    if not cleaned:
        return []
    tokens = tokenize_ids(cleaned)
    if not tokens:
        return []
    candidates: List[str] = []
    idx = 0
    while idx < len(tokens):
        node, next_idx = parse_ids_from_tokens(tokens, idx)
        candidates.append(node_surface(node))
        idx = next_idx
    return candidates


def parse_ids_string(text: str) -> IDSNode:
    candidates = extract_ids_candidates(text)
    if not candidates:
        raise ParseError(f"Could not parse IDS: {text!r}")
    tokens = tokenize_ids(candidates[0])
    node, next_idx = parse_ids_from_tokens(tokens, 0)
    if next_idx != len(tokens):
        raise ParseError(f"Trailing tokens in IDS: {text!r}")
    return node


def parse_chise_like_file(path: Path, source_format: str) -> Dict[str, DecompositionEntry]:
    """
    Parser for CHISE-like or cjkvi-like text files.

    Supported line styles:
      U+XXXX<TAB>字<TAB>IDS
      U+XXXX<TAB>字<TAB>IDS<TAB>@apparent=IDS
      U+XXXX 字 IDS IDS2
    """
    out: Dict[str, DecompositionEntry] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(";;"):
                continue
            m = CHISE_LIKE_LEAD_RE.match(raw_line.rstrip("\n"))
            if not m:
                # Some files contain standalone metadata or malformed lines; skip them.
                continue
            cp_token = m.group("cp")
            char_field = m.group("char")
            rest = m.group("rest")
            char_from_cp = char_from_codepoint_token(cp_token)
            char = char_field
            if char_from_cp is not None and char != char_from_cp:
                # Prefer the explicit character column unless it is suspiciously empty.
                if len(char) != 1:
                    char = char_from_cp

            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                tail_fields = [field.strip() for field in fields[2:] if field.strip()]
            else:
                tail_fields = [rest]

            entry = DecompositionEntry(key=char, char=char, source_format=source_format, source_lines=[lineno])

            ids_candidates: List[str] = []
            apparent_candidates: List[str] = []
            metadata: Dict[str, List[str]] = defaultdict(list)

            for field in tail_fields:
                if not field:
                    continue
                if field.startswith("@apparent="):
                    try:
                        apparent_candidates.extend(extract_ids_candidates(field[len("@apparent=") :]))
                    except ParseError:
                        metadata["unparsed_apparent"].append(field[len("@apparent=") :].strip())
                    continue
                if field.startswith("@"):
                    key, eq, value = field.partition("=")
                    meta_key = key[1:] if key.startswith("@") else key
                    metadata[meta_key].append(value if eq else "")
                    continue
                try:
                    ids_candidates.extend(extract_ids_candidates(field))
                except ParseError:
                    metadata["unparsed_field"].append(field)

            if not ids_candidates:
                # If the tab-split path did not yield IDS candidates, try the whole tail.
                try:
                    ids_candidates = extract_ids_candidates(rest)
                except ParseError:
                    metadata["unparsed_tail"].append(rest)
                    ids_candidates = []

            if ids_candidates:
                entry.primary_raw = ids_candidates[0]
                entry.primary_tree = parse_ids_string(ids_candidates[0])
                entry.alternative_raw = dedupe_preserve_order(ids_candidates[1:])
            entry.apparent_raw = dedupe_preserve_order(apparent_candidates)
            entry.metadata = metadata

            existing = out.get(entry.key)
            if existing is None:
                out[entry.key] = entry
            else:
                existing.merge_from(entry)
    return out


def split_cjk_decomp_parts(parts_text: str) -> List[str]:
    if not parts_text.strip():
        return []
    return [part.strip() for part in parts_text.split(",") if part.strip()]


def parse_cjk_decomp_file(path: Path) -> Dict[str, DecompositionEntry]:
    out: Dict[str, DecompositionEntry] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = CJK_DECOMP_RE.match(line)
            if not m:
                continue
            target = m.group("target").strip()
            kind = m.group("kind").strip()
            parts = split_cjk_decomp_parts(m.group("parts"))
            char = target if len(target) == 1 else None

            entry = DecompositionEntry(key=target, char=char, source_format="cjk-decomp", source_lines=[lineno])

            if kind == "c" or not parts:
                entry.primary_raw = target
                entry.primary_tree = IDSNode("leaf", target, (), "cjk-decomp")
            else:
                children = tuple(IDSNode("leaf", part, (), "cjk-decomp") for part in parts)
                entry.primary_raw = f"{kind}({','.join(parts)})"
                entry.primary_tree = IDSNode("op", kind, children, "cjk-decomp")
            entry.metadata = defaultdict(list, {"kind": [kind]})
            out[target] = entry
    return out


def load_decomposition_db(path: Path, source_format: str) -> Dict[str, DecompositionEntry]:
    if source_format == "auto":
        # very small sniff: cjk-decomp uses "X:kind(...)" at line start
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith(";;"):
                    continue
                if CJK_DECOMP_RE.match(line):
                    return parse_cjk_decomp_file(path)
                return parse_chise_like_file(path, "auto")
        return {}
    if source_format == "cjk-decomp":
        return parse_cjk_decomp_file(path)
    return parse_chise_like_file(path, source_format)


def resolve_token_to_lookup_key(
    token: str,
    name_lookup: Dict[str, str],
    equivalent_map: Dict[str, str],
    entity_map: Dict[str, str],
) -> str:
    if token in entity_map:
        return entity_map[token]
    body = token[1:-1] if token.startswith("&") and token.endswith(";") else token
    if body in entity_map:
        return entity_map[body]

    cp_char = char_from_codepoint_token(token)
    if cp_char is not None:
        token = cp_char

    if token.startswith("&") and token.endswith(";"):
        body = token[1:-1]
        if body in entity_map:
            token = entity_map[body]
        else:
            m = U_TOKEN_ANYWHERE_RE.search(body)
            if m:
                cp_hex = m.group(1) or m.group(2)
                if cp_hex is not None:
                    cp = int(cp_hex, 16)
                    if cp <= 0x10FFFF:
                        token = chr(cp)

    if token.startswith("<") and token.endswith(">"):
        name_key = normalize_ucd_name(token[1:-1])
        replacement = name_lookup.get(name_key)
        if replacement:
            token = replacement

    token = equivalent_map.get(token, token)
    return token


def normalize_leaf_token(
    token: str,
    context: LeafContext,
    normalization: str,
    name_lookup: Dict[str, str],
    equivalent_map: Dict[str, str],
    entity_map: Dict[str, str],
) -> str:
    token = resolve_token_to_lookup_key(token, name_lookup, equivalent_map, entity_map)

    if token == "阝":
        if context.parent_op in {"⿰", "⿲"} and context.child_index is not None and context.sibling_count is not None:
            if context.child_index == context.sibling_count - 1:
                token = "邑"
            else:
                token = "阜"
        elif context.scheme == "cjk-decomp" and context.parent_op is not None:
            if context.parent_op.startswith("a") and context.child_index is not None and context.sibling_count is not None:
                if context.child_index == context.sibling_count - 1:
                    token = "邑"
                else:
                    token = "阜"

    if normalization in {"conservative", "aggressive"}:
        token = CONSERVATIVE_VARIANT_MAP.get(token, token)

    if normalization == "aggressive":
        token = AGGRESSIVE_EXTRA_VARIANT_MAP.get(token, token)

    return token


def expand_node(
    node: IDSNode,
    decomp_db: Dict[str, DecompositionEntry],
    name_lookup: Dict[str, str],
    equivalent_map: Dict[str, str],
    entity_map: Dict[str, str],
    max_depth: int,
    memo: Dict[str, IDSNode],
    visiting: Optional[set] = None,
    depth: int = 0,
) -> IDSNode:
    if visiting is None:
        visiting = set()

    if depth >= max_depth:
        return node

    if node.is_leaf:
        lookup_key = resolve_token_to_lookup_key(node.value, name_lookup, equivalent_map, entity_map)
        if lookup_key in memo:
            return memo[lookup_key]
        if lookup_key in visiting:
            return IDSNode("leaf", lookup_key, (), node.scheme)
        entry = decomp_db.get(lookup_key)
        if entry is None or entry.primary_tree is None:
            return IDSNode("leaf", lookup_key, (), node.scheme)
        if entry.primary_tree.is_leaf and resolve_token_to_lookup_key(entry.primary_tree.value, name_lookup, equivalent_map, entity_map) == lookup_key:
            return IDSNode("leaf", lookup_key, (), node.scheme)

        visiting.add(lookup_key)
        expanded = expand_node(
            entry.primary_tree,
            decomp_db,
            name_lookup,
            equivalent_map,
            entity_map,
            max_depth,
            memo,
            visiting,
            depth + 1,
        )
        visiting.remove(lookup_key)
        memo[lookup_key] = expanded
        return expanded

    expanded_children = tuple(
        expand_node(
            child,
            decomp_db,
            name_lookup,
            equivalent_map,
            entity_map,
            max_depth,
            memo,
            visiting,
            depth + 1,
        )
        for child in node.children
    )
    return IDSNode("op", node.value, expanded_children, node.scheme)


def build_record(cp: int, ctx: BuildContext) -> dict:
    char = chr(cp)
    name = ctx.assigned_main_block[cp]
    unihan_props = ctx.unihan.get(cp, {})
    rs_values = parse_rs_unicode(unihan_props.get("kRSUnicode"), ctx.radical_map)
    total_strokes = parse_total_strokes(unihan_props.get("kTotalStrokes"))
    definition = unihan_props.get("kDefinition")

    entry = ctx.decomp_db.get(char)
    primary_raw = entry.primary_raw if entry else None
    primary_tree = entry.primary_tree if entry else None
    immediate_components = immediate_component_surfaces(primary_tree) if primary_tree else []
    alt_raw = entry.alternative_raw if entry else []
    apparent_raw = entry.apparent_raw if entry else []
    metadata = entry.metadata if entry else {}
    source_lines = entry.source_lines if entry else []

    expanded_tree = None
    raw_leaf_tokens: List[str] = []
    normalized_leaf_tokens: List[str] = []
    unresolved_tokens: List[str] = []
    if primary_tree is not None:
        expanded_tree = expand_node(
            primary_tree,
            ctx.decomp_db,
            ctx.name_lookup,
            ctx.equivalent_map,
            ctx.entity_map,
            ctx.max_depth,
            memo={},
        )
        raw_leaf_pairs = collect_leaf_tokens(expanded_tree)
        raw_leaf_tokens = [resolve_token_to_lookup_key(tok, ctx.name_lookup, ctx.equivalent_map, ctx.entity_map) for tok, _ in raw_leaf_pairs]
        normalized_leaf_tokens = [
            normalize_leaf_token(
                tok,
                context,
                ctx.normalization,
                ctx.name_lookup,
                ctx.equivalent_map,
                ctx.entity_map,
            )
            for tok, context in raw_leaf_pairs
        ]
        unresolved_tokens = dedupe_preserve_order(
            tok
            for tok in normalized_leaf_tokens
            if is_unresolved_token(tok)
        )

    record = {
        "char": char,
        "codepoint": cp_to_uplus(cp),
        "codepoint_int": cp,
        "name": name,
        "block": MAIN_BLOCK_NAME,
        "unihan": {
            "kRSUnicode_raw": unihan_props.get("kRSUnicode"),
            "kRSUnicode": [item.to_dict() for item in rs_values],
            "kTotalStrokes_raw": unihan_props.get("kTotalStrokes"),
            "kTotalStrokes": total_strokes,
            "kDefinition": definition,
        },
        "decomposition_source": entry.source_format if entry else None,
        "decomposition": {
            "primary_raw": primary_raw,
            "alternative_raw": alt_raw,
            "apparent_raw": apparent_raw,
            "metadata": {k: list(v) for k, v in metadata.items()},
            "source_lines": source_lines,
            "immediate_components": immediate_components,
            "primary_tree": serialize_node(primary_tree) if primary_tree else None,
            "expanded_tree": serialize_node(expanded_tree) if expanded_tree else None,
            "leaf_components_raw": raw_leaf_tokens,
            "leaf_components_raw_unique": dedupe_preserve_order(raw_leaf_tokens),
            "leaf_radicals_normalized": normalized_leaf_tokens,
            "leaf_radicals_normalized_unique": dedupe_preserve_order(normalized_leaf_tokens),
            "unresolved_tokens": unresolved_tokens,
        },
    }
    return record


def write_records(records: Iterable[dict], output: Path, output_format: str) -> None:
    ensure_dir(output.parent)
    if output_format == "jsonl":
        with output.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
                fh.write("\n")
        return
    if output_format == "json":
        with output.open("w", encoding="utf-8") as fh:
            json.dump(list(records), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return
    raise HanDecompError(f"Unsupported output format: {output_format}")


def load_records_for_lookup(dataset: Path) -> Iterator[dict]:
    suffix = dataset.suffix.lower()
    if suffix == ".jsonl":
        with dataset.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        with dataset.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                yield item
            return
    raise HanDecompError(f"Unsupported dataset format for lookup: {dataset}")


def summarize_stats(records: Iterable[dict]) -> dict:
    total = 0
    with_decomp = 0
    with_unresolved = 0
    no_unihan = 0
    for record in records:
        total += 1
        if record["decomposition"]["primary_raw"]:
            with_decomp += 1
        if record["decomposition"]["unresolved_tokens"]:
            with_unresolved += 1
        if not record["unihan"]["kRSUnicode"] and not record["unihan"]["kTotalStrokes"]:
            no_unihan += 1
    return {
        "total_records": total,
        "with_decomposition": with_decomp,
        "with_unresolved_tokens": with_unresolved,
        "without_unihan_radical_or_strokes": no_unihan,
    }


def build_dataset(
    unicode_data_path: Path,
    unihan_zip_path: Path,
    cjk_radicals_path: Path,
    equivalent_unified_path: Path,
    decomp_path: Optional[Path],
    decomp_source: str,
    entity_map_path: Optional[Path],
    normalization: str,
    output: Path,
    output_format: str,
    include_unassigned: bool,
    max_depth: int,
) -> dict:
    assigned, name_lookup = parse_unicode_data(unicode_data_path)
    radical_map = parse_cjk_radicals(cjk_radicals_path)
    equivalent_map = parse_equivalent_unified(equivalent_unified_path)
    unihan = parse_unihan_zip(unihan_zip_path)
    entity_map = load_json_mapping(entity_map_path)

    if decomp_path is None or decomp_source == "none":
        decomp_db: Dict[str, DecompositionEntry] = {}
    else:
        decomp_db = load_decomposition_db(decomp_path, decomp_source)

    ctx = BuildContext(
        assigned_main_block=assigned,
        name_lookup=name_lookup,
        radical_map=radical_map,
        equivalent_map=equivalent_map,
        unihan=unihan,
        decomp_db=decomp_db,
        entity_map=entity_map,
        normalization=normalization,
        max_depth=max_depth,
    )

    cps = range(MAIN_BLOCK_START, MAIN_BLOCK_END + 1)
    if not include_unassigned:
        cps = [cp for cp in cps if cp in assigned]

    records = [build_record(cp, ctx) for cp in cps]
    write_records(records, output, output_format)
    return summarize_stats(records)


def resolve_default_input_paths(
    data_dir: Path,
    decomp_source: str,
) -> Dict[str, Path]:
    downloads = data_dir / "downloads"
    paths = {
        "unicode_data": downloads / DOWNLOAD_FILENAMES["unicode_data"],
        "unihan_zip": downloads / DOWNLOAD_FILENAMES["unihan_zip"],
        "cjk_radicals": downloads / DOWNLOAD_FILENAMES["cjk_radicals"],
        "equivalent_unified": downloads / DOWNLOAD_FILENAMES["equivalent_unified"],
    }
    if decomp_source != "none":
        lookup_key = "cjkvi" if decomp_source == "auto" else decomp_source
        filename = DOWNLOAD_FILENAMES.get(lookup_key)
        if filename:
            paths["decomp"] = downloads / filename
    return paths


def parse_char_or_codepoint(arg: str) -> str:
    cp = parse_codepoint_token(arg)
    if cp is not None:
        return chr(cp)
    if len(arg) == 1:
        return arg
    raise HanDecompError(f"Lookup key must be a single character or U+XXXX code point: {arg}")


def cmd_download(args: argparse.Namespace) -> int:
    download_required_files(
        data_dir=Path(args.data_dir),
        decomp_source=args.decomp_source,
        force=args.force,
        decomp_url=args.decomp_url,
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    if args.download_missing:
        download_required_files(
            data_dir=data_dir,
            decomp_source=args.decomp_source,
            force=False,
            decomp_url=args.decomp_url,
        )
    default_paths = resolve_default_input_paths(data_dir, args.decomp_source)

    unicode_data = Path(args.unicode_data) if args.unicode_data else default_paths["unicode_data"]
    unihan_zip = Path(args.unihan_zip) if args.unihan_zip else default_paths["unihan_zip"]
    cjk_radicals = Path(args.cjk_radicals) if args.cjk_radicals else default_paths["cjk_radicals"]
    equivalent_unified = Path(args.equivalent_unified) if args.equivalent_unified else default_paths["equivalent_unified"]
    decomp_path = None
    if args.decomp_source != "none":
        if args.decomp_file:
            decomp_path = Path(args.decomp_file)
        else:
            decomp_path = default_paths.get("decomp")
            if decomp_path is None:
                raise HanDecompError("No decomposition file path available")

    output = Path(args.output)
    stats = build_dataset(
        unicode_data_path=unicode_data,
        unihan_zip_path=unihan_zip,
        cjk_radicals_path=cjk_radicals,
        equivalent_unified_path=equivalent_unified,
        decomp_path=decomp_path,
        decomp_source=args.decomp_source,
        entity_map_path=Path(args.entity_map) if args.entity_map else None,
        normalization=args.normalization,
        output=output,
        output_format=args.output_format,
        include_unassigned=args.include_unassigned,
        max_depth=args.max_depth,
    )
    stderr("Build complete.")
    stderr(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    wanted = {parse_char_or_codepoint(item) for item in args.items}
    found = {}
    for record in load_records_for_lookup(dataset):
        char = record.get("char")
        if char in wanted:
            found[char] = record

    missing = [item for item in wanted if item not in found]
    for char in args.items:
        normalized = parse_char_or_codepoint(char)
        if normalized in found:
            print(json.dumps(found[normalized], ensure_ascii=False, indent=2))
    if missing:
        stderr(f"Not found: {' '.join(missing)}")
        return 1
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    stats = summarize_stats(load_records_for_lookup(Path(args.dataset)))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def run_self_tests() -> None:
    # IDS parsing
    n = parse_ids_string("⿰氵可")
    assert n.value == "⿰" and len(n.children) == 2 and n.children[0].value == "氵"

    n = parse_ids_string("⿳艹⿰白勺口")
    assert n.value == "⿳" and len(n.children) == 3

    # Multiple IDS candidates in one field
    cands = extract_ids_candidates("⿰言吾 ⿰言⿱五口")
    assert cands == ["⿰言吾", "⿰言⿱五口"]

    # CHISE-like parsing, including @apparent
    sample = textwrap.dedent("""\
        U+6CB3\t河\t⿰氵可\t@apparent=⿰水可
        ;; comment
        U+8A9E\t語\t⿰言吾
    """)
    tmp = Path("._selftest_chise.txt")
    tmp.write_text(sample, encoding="utf-8")
    db = parse_chise_like_file(tmp, "chise")
    tmp.unlink()
    assert db["河"].primary_raw == "⿰氵可"
    assert db["河"].apparent_raw == ["⿰水可"]
    assert db["語"].primary_tree is not None

    # cjk-decomp parsing
    sample = textwrap.dedent("""\
        的:a(白,勺)
        00001:c()
    """)
    tmp = Path("._selftest_cjkdecomp.txt")
    tmp.write_text(sample, encoding="utf-8")
    db = parse_cjk_decomp_file(tmp)
    tmp.unlink()
    assert db["的"].primary_tree is not None
    assert db["的"].primary_tree.value == "a"
    assert db["00001"].primary_tree is not None
    assert db["00001"].primary_tree.is_leaf

    # RS Unicode parsing
    radicals = {"85": ("⽔", "水"), "170'": ("⻖", "阜")}
    values = parse_rs_unicode("85.5 170'.3", radicals)
    assert values[0].radical_number == 85
    assert values[1].simplification_marks == 1

    # Token resolution / normalization
    name_lookup = {normalize_ucd_name("CJK RADICAL MEAT"): "⺼"}
    equivalent = {"⺼": "肉"}
    entity_map = {}
    tok = normalize_leaf_token("<CJK RADICAL MEAT>", LeafContext(None, None, None, None), "conservative", name_lookup, equivalent, entity_map)
    assert tok == "肉"

    # Context-sensitive 阜/邑 resolution
    tok = normalize_leaf_token("阝", LeafContext("⿰", "ids", 0, 2), "conservative", {}, {}, {})
    assert tok == "阜"
    tok = normalize_leaf_token("阝", LeafContext("⿰", "ids", 1, 2), "conservative", {}, {}, {})
    assert tok == "邑"

    # Recursive expansion with synthetic DB
    river = DecompositionEntry(key="河", char="河", source_format="ids", primary_raw="⿰氵可", primary_tree=parse_ids_string("⿰氵可"))
    comp = DecompositionEntry(key="可", char="可", source_format="ids", primary_raw="⿱丁口", primary_tree=parse_ids_string("⿱丁口"))
    db = {"河": river, "可": comp}
    expanded = expand_node(river.primary_tree, db, {}, {}, {}, 16, memo={})
    leaves = [tok for tok, _ in collect_leaf_tokens(expanded)]
    assert leaves == ["氵", "丁", "口"]

    # Node surface for cjk-decomp
    node = IDSNode("op", "a", (IDSNode("leaf", "白", (), "cjk-decomp"), IDSNode("leaf", "勺", (), "cjk-decomp")), "cjk-decomp")
    assert node_surface(node) == "a(白,勺)"

    stderr("All self-tests passed.")


def cmd_self_test(args: argparse.Namespace) -> int:
    run_self_tests()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build recursive Han subcomponent decompositions for U+4E00..U+9FFF."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download the default Unicode/decomposition source files.")
    p_download.add_argument("--data-dir", default="data", help="Directory used for cached downloads.")
    p_download.add_argument(
        "--decomp-source",
        default="cjkvi",
        choices=("cjkvi", "chise", "cjk-decomp", "none"),
        help="Decomposition source to download.",
    )
    p_download.add_argument("--decomp-url", default=None, help="Override the decomposition source URL.")
    p_download.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    p_download.set_defaults(func=cmd_download)

    p_build = subparsers.add_parser("build", help="Build the main-block dataset.")
    p_build.add_argument("--data-dir", default="data", help="Directory used for cached downloads.")
    p_build.add_argument(
        "--decomp-source",
        default="cjkvi",
        choices=("cjkvi", "chise", "cjk-decomp", "auto", "none"),
        help="Which decomposition parser/source style to use.",
    )
    p_build.add_argument("--decomp-url", default=None, help="Override the download URL used with --download-missing.")
    p_build.add_argument("--download-missing", action="store_true", help="Auto-download missing default files.")
    p_build.add_argument("--unicode-data", default=None, help="Path to UnicodeData.txt")
    p_build.add_argument("--unihan-zip", default=None, help="Path to Unihan.zip")
    p_build.add_argument("--cjk-radicals", default=None, help="Path to CJKRadicals.txt")
    p_build.add_argument("--equivalent-unified", default=None, help="Path to EquivalentUnifiedIdeograph.txt")
    p_build.add_argument("--decomp-file", default=None, help="Path to a decomposition file.")
    p_build.add_argument("--entity-map", default=None, help="Optional JSON file mapping unresolved entities/tokens to Unicode.")
    p_build.add_argument(
        "--normalization",
        default="conservative",
        choices=("none", "conservative", "aggressive"),
        help="How aggressively to normalize final leaf components into radicals.",
    )
    p_build.add_argument("--max-depth", type=int, default=32, help="Maximum recursive expansion depth.")
    p_build.add_argument("--include-unassigned", action="store_true", help="Include unassigned code points in U+4E00..U+9FFF.")
    p_build.add_argument("--output", required=True, help="Output file path.")
    p_build.add_argument("--output-format", default="jsonl", choices=("jsonl", "json"), help="Output serialization format.")
    p_build.set_defaults(func=cmd_build)

    p_lookup = subparsers.add_parser("lookup", help="Lookup one or more characters/code points from a built dataset.")
    p_lookup.add_argument("--dataset", required=True, help="Path to a .jsonl or .json dataset created by this script.")
    p_lookup.add_argument("items", nargs="+", help="Characters or U+XXXX code points.")
    p_lookup.set_defaults(func=cmd_lookup)

    p_stats = subparsers.add_parser("stats", help="Summarize a built dataset.")
    p_stats.add_argument("--dataset", required=True, help="Path to a .jsonl or .json dataset.")
    p_stats.set_defaults(func=cmd_stats)

    p_self = subparsers.add_parser("self-test", help="Run internal parser/expansion self-tests.")
    p_self.set_defaults(func=cmd_self_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HanDecompError as exc:
        stderr(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        stderr("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

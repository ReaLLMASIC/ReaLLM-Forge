#!/usr/bin/env python3
"""
verilog_ts_colorize.py

Tree-sitter Verilog highlight query -> per-byte 1-char symbol stream.

Targets your environment:
- tree_sitter==0.25.2 (QueryCursor API; Parser.language expects Language)
- tree-sitter-verilog==1.0.3 (language() returns PyCapsule)

Robust query compatibility:
1) Split highlights.scm into top-level S-expressions.
2) Drop rules referencing unknown node types.
3) Drop rules referencing unknown field names.
4) Drop rules that fail to compile alone (fixes "Impossible pattern").
5) Run the final query using QueryCursor variants.

Output:
- A shadow file with EXACTLY the same byte length as the source file.
- Newlines preserved.
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from tree_sitter import Language, Parser, Query
from tree_sitter_verilog import language as verilog_language

# QueryCursor in tree_sitter 0.25.x exists, but constructor/signatures vary.
try:
    from tree_sitter import QueryCursor
except Exception:
    QueryCursor = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Span:
    start: int
    end: int
    capture: str


# -----------------------------
# IO helpers
# -----------------------------

def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def load_text_utf8(path: str) -> str:
    return load_bytes(path).decode("utf-8")


def load_symbol_map(path: Optional[str]) -> Dict[str, str]:
    default = {
        "comment": "c",
        "string": "s",
        "keyword": "k",
        "type": "t",
        "number": "n",
        "constant": "o",
        "function": "f",
        "method": "m",
        "property": "p",
        "variable": "v",
        "parameter": "a",
        "operator": "x",
        "punctuation": ".",
        "attribute": "u",
        "label": "l",
        "module": "M",
        "namespace": "N",
        "tag": "g",
        "escape": "e",
        "boolean": "b",
        "field": "F",
        "identifier": "i",
        "default": "_",
    }
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        user_map = json.load(f)
    default.update(user_map)
    return default


# -----------------------------
# Tree-sitter compatibility
# -----------------------------

def get_verilog_ts_language() -> Language:
    """
    tree_sitter_verilog.language is typically a callable returning PyCapsule.
    tree_sitter.Language(capsule) wraps it into a Language instance.
    """
    cap_or_lang = verilog_language() if callable(verilog_language) else verilog_language
    if isinstance(cap_or_lang, Language):
        return cap_or_lang
    return Language(cap_or_lang)


def set_parser_language(parser: Parser, lang: Language) -> None:
    """
    tree_sitter 0.25.x uses parser.language = lang
    Some older builds have parser.set_language(lang)
    """
    if hasattr(parser, "set_language"):
        parser.set_language(lang)  # type: ignore[attr-defined]
    else:
        parser.language = lang  # type: ignore[attr-defined]


# -----------------------------
# Query execution compatibility
# -----------------------------

def capture_name_from_id(query: Query, cap_id: int) -> str:
    if hasattr(query, "capture_name_for_id"):
        return query.capture_name_for_id(cap_id)  # type: ignore[attr-defined]
    # Some builds expose capture_names
    return query.capture_names[cap_id]  # type: ignore[attr-defined]


def iter_query_captures(query: Query, root_node, source_bytes: bytes) -> Iterable[Tuple[object, str]]:
    """
    Normalize query captures across API variants to yield (node, capture_name).
    """
    # Very old API (not yours, but harmless)
    if hasattr(query, "captures"):
        for node, cap in query.captures(root_node):  # type: ignore[attr-defined]
            yield node, cap
        return

    if QueryCursor is None:
        raise RuntimeError("QueryCursor unavailable and Query.captures missing; cannot execute query.")

    # Constructor variants:
    #   cursor = QueryCursor()            then cursor.captures(query, node, src)
    #   cursor = QueryCursor(query)      then cursor.captures(node, src)
    try:
        cursor = QueryCursor()  # type: ignore[call-arg]
        bound_query = False
    except TypeError:
        cursor = QueryCursor(query)  # type: ignore[call-arg]
        bound_query = True

    # Call variants for captures:
    # Try a small matrix of possibilities deterministically.
    def try_captures():
        # returns iterable of (node, cap_id_or_name)
        if not bound_query:
            # cursor.captures(query, node, src) OR cursor.captures(query, node)
            try:
                return cursor.captures(query, root_node, source_bytes)
            except TypeError:
                return cursor.captures(query, root_node)
        else:
            # cursor.captures(node, src) OR cursor.captures(node)
            try:
                return cursor.captures(root_node, source_bytes)
            except TypeError:
                return cursor.captures(root_node)

    results = try_captures()

    for item in results:
        # Some bindings yield 2-tuples, others 3-tuples (or more).
        if not isinstance(item, tuple):
            continue

        node = None
        cap = None

        # Find the "node-like" element (has start_byte/end_byte)
        for x in item:
            if hasattr(x, "start_byte") and hasattr(x, "end_byte"):
                node = x
                break

        # Find the "capture-like" element (str or int)
        for x in item:
            if isinstance(x, str) or isinstance(x, int):
                cap = x
                # Don't accidentally pick up an integer that is actually a byte offset;
                # capture ids are usually small, but if you want you can add heuristics here.
                break

        if node is None or cap is None:
            # If we couldn't confidently parse this tuple, skip it.
            continue

        if isinstance(cap, str):
            yield node, cap
        else:
            yield node, capture_name_from_id(query, int(cap))



# -----------------------------
# Query filtering (robust)
# -----------------------------

def language_node_types(lang: Language) -> set[str]:
    names = set()
    for i in range(1, lang.node_kind_count + 1):
        n = lang.node_kind_for_id(i)
        if n:
            names.add(n)
    return names


def language_field_names(lang: Language) -> set[str]:
    names = set()
    for i in range(1, lang.field_count + 1):
        n = lang.field_name_for_id(i)
        if n:
            names.add(n)
    return names


def split_top_level_sexps(src: str) -> List[str]:
    """
    Best-effort split into top-level S-expressions, respecting strings.
    """
    sexps: List[str] = []
    depth = 0
    start = None
    in_str = False
    esc = False

    for i, ch in enumerate(src):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue

        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                sexps.append(src[start:i + 1])
                start = None

    return sexps


def sexp_mentions_unknown_syntax(sexp: str, known_nodes: set[str], known_fields: set[str]) -> bool:
    """
    Drop rules referencing unknown node types or field names.
    """
    no_strings = re.sub(r'"([^"\\]|\\.)*"', '""', sexp)

    # field refs like port_name:
    for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:", no_strings):
        field = m.group(1)
        if field not in known_fields:
            return True

    # node refs like (config ...) or ( module_declaration ...)
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", no_strings)
    ignore = {
        "eq", "match", "any_of", "all_of", "not_any_of", "not_eq",
        "set", "set!", "is", "is?", "has_ancestor", "has_ancestor?",
        "has_parent", "has_parent?", "contains", "contains?",
        "lua_match", "lua-match?", "vim_match", "vim-match?",
        "conceal", "offset", "offset!", "make_range", "make-range!",
        "injection", "language", "content",
        "true", "false", "nil",
    }

    for t in tokens:
        if t in ignore:
            continue
        if t not in known_nodes:
            if f"({t}" in sexp or f"( {t}" in sexp:
                return True

    return False


def compile_safe_filter(sexps: List[str], lang: Language, verbose: bool = False) -> Tuple[List[str], int]:
    kept: List[str] = []
    dropped = 0
    for idx, sexp in enumerate(sexps):
        try:
            _ = Query(lang, sexp)
            kept.append(sexp)
        except Exception as e:
            dropped += 1
            if verbose:
                msg = str(e).splitlines()[0] if str(e) else e.__class__.__name__
                print(f"[query-compile-drop] rule#{idx} dropped: {msg}")
    return kept, dropped


def filter_query_robust(query_src: str, lang: Language, verbose: bool = False) -> str:
    sexps = split_top_level_sexps(query_src)
    known_nodes = language_node_types(lang)
    known_fields = language_field_names(lang)

    step2: List[str] = []
    dropped_unknown = 0
    for s in sexps:
        if sexp_mentions_unknown_syntax(s, known_nodes, known_fields):
            dropped_unknown += 1
        else:
            step2.append(s)

    step3, dropped_compile = compile_safe_filter(step2, lang, verbose=verbose)

    if verbose:
        print(f"[query-filter] total={len(sexps)} kept={len(step3)} "
              f"dropped_unknown={dropped_unknown} dropped_compile={dropped_compile}")

    return "\n\n".join(step3) + "\n"


# -----------------------------
# Highlight -> per-byte symbols
# -----------------------------

def family_from_capture(name: str) -> str:
    if name.startswith("@"):
        name = name[1:]
    return name.split(".")[0] if name else "default"


def build_spans(tree, query: Query, source_bytes: bytes) -> List[Span]:
    spans: List[Span] = []
    for node, cap_name in iter_query_captures(query, tree.root_node, source_bytes):
        spans.append(Span(start=node.start_byte, end=node.end_byte, capture=cap_name))
    return spans


def apply_spans_to_bytes(
    source_bytes: bytes,
    spans: List[Span],
    sym_map: Dict[str, str],
    prefer_longest: bool = True,
    preserve_newlines: bool = True,
) -> bytes:
    default_sym = sym_map.get("default", "_")
    if len(default_sym) != 1:
        raise ValueError("symbol-map 'default' must be exactly 1 character")

    # CRITICAL: output must match source length
    out = bytearray(len(source_bytes))
    fill_byte = default_sym.encode("ascii", errors="replace")[0]
    out[:] = bytes([fill_byte]) * len(out)

    # Safety: never silently output wrong size
    if len(out) != len(source_bytes):
        raise RuntimeError(f"Length mismatch constructing out: out={len(out)} source={len(source_bytes)}")

    if prefer_longest:
        spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start, s.end))

    for sp in spans:
        fam = family_from_capture(sp.capture)
        sym = sym_map.get(sp.capture) or sym_map.get(fam) or default_sym
        if len(sym) != 1:
            raise ValueError(f"Symbol for '{sp.capture}'/'{fam}' must be 1 char, got: {sym!r}")
        sym_b = sym.encode("ascii", errors="replace")[0]

        start = max(0, sp.start)
        end = min(len(out), sp.end)
        for i in range(start, end):
            if out[i] == fill_byte:
                out[i] = sym_b

    if preserve_newlines:
        # Defensive clamp (should be equal anyway)
        n = min(len(out), len(source_bytes))
        for i in range(n):
            b = source_bytes[i]
            if b in (0x0A, 0x0D):  # \n or \r
                out[i] = b

    return bytes(out)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Tree-sitter Verilog highlights -> per-byte symbol shadow file")
    ap.add_argument("verilog_file", help="Input .v/.sv file")
    ap.add_argument("--highlights", required=True, help="Tree-sitter highlight query file (highlights.scm)")
    ap.add_argument("-o", "--out", default=None, help="Output file (default: <input>.tscolors.txt)")
    ap.add_argument("--symbol-map", default=None, help="JSON mapping capture/family -> 1-char symbol")
    ap.add_argument("--prefer-longest", action="store_true", help="Prefer longest spans when overlaps exist")
    ap.add_argument("--no-preserve-newlines", action="store_true", help="Do not preserve newline bytes in output")
    ap.add_argument("--keep-unfiltered-query", action="store_true",
                    help="Do not filter highlights.scm (may error if mismatch exists)")
    ap.add_argument("--verbose-filter", action="store_true",
                    help="Print kept/dropped counts and per-rule drops during filtering")
    args = ap.parse_args()

    src = load_bytes(args.verilog_file)
    sym_map = load_symbol_map(args.symbol_map)

    lang = get_verilog_ts_language()
    parser = Parser()
    set_parser_language(parser, lang)
    tree = parser.parse(src)

    qsrc = load_text_utf8(args.highlights)
    if not args.keep_unfiltered_query:
        qsrc = filter_query_robust(qsrc, lang, verbose=args.verbose_filter)

    query = Query(lang, qsrc)

    spans = build_spans(tree, query, src)

    colored = apply_spans_to_bytes(
        src,
        spans,
        sym_map,
        prefer_longest=args.prefer_longest,
        preserve_newlines=not args.no_preserve_newlines,
    )

    out_path = args.out or (args.verilog_file + ".tscolors.txt")
    with open(out_path, "wb") as f:
        f.write(colored)

    # Safety print
    print(f"Wrote: {out_path} ({len(colored)} bytes)")
    if len(colored) != len(src):
        raise RuntimeError(f"Output length mismatch: src={len(src)} out={len(colored)}")


if __name__ == "__main__":
    main()


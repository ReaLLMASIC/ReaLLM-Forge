#!/usr/bin/env python3
"""compare_meta_vocab_tui.py

Textual TUI to compare vocabularies stored in two meta.pkl files.
Supports sorting both vocab lists by byte length or frequency.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static


@dataclass(frozen=True)
class VocabEntry:
    token_id: int
    token: Any
    byte_len: int
    count: int


def _load_meta(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_meta_path(path: Path) -> Path:
    if path.is_dir():
        return path / "meta.pkl"
    return path


def _token_byte_len(token: Any) -> int:
    if isinstance(token, bytes):
        return len(token)
    if isinstance(token, str):
        return len(token.encode("utf-8"))
    return len(str(token).encode("utf-8"))


def _display_token(token: Any) -> str:
    if isinstance(token, bytes):
        return repr(token)
    return repr(token)


def _coerce_token_id(raw_id: Any) -> int:
    if isinstance(raw_id, int):
        return raw_id
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return -1


def _build_entries(meta: dict) -> List[VocabEntry]:
    itos = meta.get("itos", {})
    token_counts = meta.get("token_counts", {}) or {}
    entries: List[VocabEntry] = []
    for raw_id, token in itos.items():
        token_id = _coerce_token_id(raw_id)
        count = int(token_counts.get(raw_id, token_counts.get(token_id, 0)) or 0)
        entries.append(
            VocabEntry(
                token_id=token_id,
                token=token,
                byte_len=_token_byte_len(token),
                count=count,
            )
        )
    return entries


def _sorted_entries(entries: Iterable[VocabEntry], mode: str) -> List[VocabEntry]:
    if mode == "bytes":
        return sorted(entries, key=lambda e: (e.byte_len, e.token_id), reverse=True)
    if mode == "freq":
        return sorted(entries, key=lambda e: (e.count, e.token_id), reverse=True)
    return sorted(entries, key=lambda e: e.token_id)


class VocabCompareApp(App):
    CSS = """
    #main { height: 1fr; }
    .panel { height: 1fr; width: 1fr; }
    DataTable { height: 1fr; width: 1fr; }
    """

    BINDINGS = [
        Binding("b", "sort_bytes", "Sort by bytes", show=True),
        Binding("f", "sort_freq", "Sort by frequency", show=True),
        Binding("i", "sort_id", "Sort by id", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, left_meta: Path, right_meta: Path) -> None:
        super().__init__()
        self.left_meta_path = left_meta
        self.right_meta_path = right_meta
        self.left_meta = _load_meta(left_meta)
        self.right_meta = _load_meta(right_meta)
        self.left_entries = _build_entries(self.left_meta)
        self.right_entries = _build_entries(self.right_meta)
        self.sort_mode = "id"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(classes="panel"):
                yield Static(self._panel_title(self.left_meta_path, self.left_meta), id="left_title")
                yield DataTable(id="left_table", zebra_stripes=True)
            with Vertical(classes="panel"):
                yield Static(self._panel_title(self.right_meta_path, self.right_meta), id="right_title")
                yield DataTable(id="right_table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        self._setup_table(self.query_one("#left_table", DataTable))
        self._setup_table(self.query_one("#right_table", DataTable))
        self._refresh_tables()

    def _panel_title(self, path: Path, meta: dict) -> str:
        tokenizer = meta.get("tokenizer", "unknown")
        vocab_size = meta.get("vocab_size", "?")
        return f"{path} | {tokenizer} | vocab={vocab_size}"

    def _setup_table(self, table: DataTable) -> None:
        self._ensure_columns(table)
        table.clear()

    def _ensure_columns(self, table: DataTable) -> None:
        if table.columns:
            return
        table.add_column("id", key="id")
        table.add_column("token", key="token")
        table.add_column("bytes", key="bytes")
        table.add_column("count", key="count")

    def _refresh_tables(self) -> None:
        left_table = self.query_one("#left_table", DataTable)
        right_table = self.query_one("#right_table", DataTable)
        self._fill_table(left_table, _sorted_entries(self.left_entries, self.sort_mode))
        self._fill_table(right_table, _sorted_entries(self.right_entries, self.sort_mode))

    def _fill_table(self, table: DataTable, entries: List[VocabEntry]) -> None:
        self._ensure_columns(table)
        table.clear()
        for entry in entries:
            table.add_row(
                str(entry.token_id),
                _display_token(entry.token),
                str(entry.byte_len),
                str(entry.count),
            )

    def action_sort_bytes(self) -> None:
        self.sort_mode = "bytes"
        self._refresh_tables()

    def action_sort_freq(self) -> None:
        self.sort_mode = "freq"
        self._refresh_tables()

    def action_sort_id(self) -> None:
        self.sort_mode = "id"
        self._refresh_tables()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vocabularies from two meta.pkl files.")
    parser.add_argument("left", type=Path, help="Path to left meta.pkl or directory containing it")
    parser.add_argument("right", type=Path, help="Path to right meta.pkl or directory containing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_meta = _resolve_meta_path(args.left)
    right_meta = _resolve_meta_path(args.right)
    if not left_meta.exists():
        raise FileNotFoundError(f"Missing meta.pkl at {left_meta}")
    if not right_meta.exists():
        raise FileNotFoundError(f"Missing meta.pkl at {right_meta}")
    app = VocabCompareApp(left_meta=left_meta, right_meta=right_meta)
    app.run()


if __name__ == "__main__":
    main()


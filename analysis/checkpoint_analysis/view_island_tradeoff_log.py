#!/usr/bin/env python3
"""TUI viewer for island tradeoff search_log.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static

DEFAULT_LOG = "search_log.yaml"
POLL_INTERVAL = 3.0
SUMMARY_LABEL = "Selected Summary"
SELECTED_STYLE = "bold orange3"


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def heat_text(value: float, lo: float, hi: float) -> Text:
    if hi <= lo:
        t = 0.0
    else:
        t = (value - lo) / (hi - lo)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    style = f"rgb({r},{g},0)"
    return Text(f"{value:.6f}", style=style)


class IslandTradeoffViewer(App):
    CSS = """#navbox{width:24;} #main{width:1fr;}"""
    BINDINGS = [Binding("q", "quit", show=False)]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self.rounds: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.idx = 0
        self.show_summary = False
        self._mtime = 0.0

    def compose(self):  # type: ignore[override]
        yield Header()
        with Horizontal():
            with Vertical(id="navbox"):
                yield Static("Rounds", classes="title")
                yield DataTable(id="nav", show_header=False, zebra_stripes=True)
            with Vertical(id="main"):
                yield Static(id="top")
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    def on_mount(self):  # type: ignore[override]
        self._reload(initial=True)
        self._build_nav()
        self._refresh()
        self.set_interval(POLL_INTERVAL, self._poll)

    def _poll(self):
        mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0
        if mtime != self._mtime:
            remember = self.idx
            self._reload(initial=False)
            self.idx = min(remember, max(0, len(self.rounds) - 1))
            self._build_nav()
            self._refresh()

    def _reload(self, initial: bool):
        data = load_yaml(self.log_path)
        self.rounds = data.get("rounds", [])
        self.summary = {
            "dataset": data.get("dataset", "-"),
            "baseline": data.get("baseline", {}),
            "search": data.get("search", {}),
            "selected": data.get("selected", {}),
            "speed_comparison": data.get("speed_comparison", {}),
        }
        if initial and self.rounds:
            self.idx = len(self.rounds) - 1
        self._mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0

    def _build_nav(self):
        nav = self.query_one("#nav", DataTable)
        nav.clear(columns=True)
        nav.add_column("item")
        for r in self.rounds:
            nav.add_row(f"round {r.get('round', '?')}")
        nav.add_row(SUMMARY_LABEL)
        nav.cursor_type = "row"
        nav.cursor_coordinate = (len(self.rounds) if self.show_summary else self.idx, 0)
        nav.focus()

    def on_data_table_row_highlighted(self, e: DataTable.RowHighlighted):  # type: ignore[override]
        if e.data_table.id != "nav":
            return
        row = e.cursor_row  # type: ignore[attr-defined]
        self.show_summary = row == len(self.rounds)
        if not self.show_summary:
            self.idx = row
        self._refresh()

    def _refresh(self):
        top = self.query_one("#top", Static)
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)

        if self.show_summary:
            sel = self.summary.get("selected", {})
            base = self.summary.get("baseline", {})
            srch = self.summary.get("search", {})
            spd = self.summary.get("speed_comparison", {})
            panel_tbl = Table.grid(padding=1)
            panel_tbl.add_column(justify="right")
            panel_tbl.add_column()
            panel_tbl.add_row("Dataset", str(self.summary.get("dataset", "-")))
            panel_tbl.add_row("Baseline val", f"{base.get('val_loss', '-')}")
            panel_tbl.add_row("Max allowed", f"{base.get('max_allowed_val_loss', '-')}")
            panel_tbl.add_row("Selected val", f"{sel.get('val_loss', '-')}")
            panel_tbl.add_row("Loss delta %", f"{sel.get('loss_delta_pct', '-')}")
            panel_tbl.add_row("Decode proxy", f"{sel.get('decode_reduction_proxy', '-')}")
            panel_tbl.add_row("Selected tok ms", f"{sel.get('decode_token_latency_ms', '-')}")
            panel_tbl.add_row("Selected tok/s", f"{sel.get('decode_tokens_per_s', '-')}")
            panel_tbl.add_row("Speedup tok/s", f"{spd.get('decode_tokens_per_s_speedup', '-')}")
            panel_tbl.add_row("Accepted rounds", f"{srch.get('accepted_rounds', '-')}")
            top.update(Panel(panel_tbl, title="Selected summary", border_style="green"))

            table.add_column("tensor")
            table.add_column("num_islands")
            table.add_column("threshold")
            for tname, cfg in sorted(sel.get("config", {}).items()):
                table.add_row(tname, str(cfg.get("num_islands", "-")), str(cfg.get("threshold", "-")))
            return

        if not self.rounds:
            top.update("No rounds in log yet.")
            return

        r = self.rounds[self.idx]
        panel_tbl = Table.grid(padding=1)
        panel_tbl.add_column(justify="right")
        panel_tbl.add_column()
        panel_tbl.add_row("Round", str(r.get("round", "-")))
        panel_tbl.add_row("Pre-round val", str(r.get("pre_round_val_loss", "-")))
        panel_tbl.add_row("Post-round val", str(r.get("post_round_val_loss", "-")))
        panel_tbl.add_row("Current tok ms", str(r.get("current_decode_token_latency_ms", "-")))
        panel_tbl.add_row("Current tok/s", str(r.get("current_decode_tokens_per_s", "-")))
        panel_tbl.add_row("Stop reason", str(r.get("stop_reason", "-")))
        best = r.get("best_candidate") or {}
        panel_tbl.add_row("Best tensor", str(best.get("tensor", "-")))
        panel_tbl.add_row("Best Δloss%", str(best.get("loss_delta_pct", "-")))
        sel = r.get("selected") or {}
        panel_tbl.add_row("Selected tensor", str(sel.get("tensor", "-")))
        panel_tbl.add_row("Selected val", str(sel.get("val_loss", "-")))
        top.update(Panel(panel_tbl, title="Round detail", border_style="cyan"))

        table.add_column("tensor")
        table.add_column("from_thr")
        table.add_column("to_thr")
        table.add_column("from_islands")
        table.add_column("to_islands")
        table.add_column("val_loss")
        table.add_column("loss_delta_pct")
        table.add_column("tok_ms")
        table.add_column("tok/s")
        table.add_column("decode_proxy")
        table.add_column("feasible")

        tested = r.get("tested", [])
        vals = [float(c.get("val_loss", 0.0)) for c in tested if isinstance(c.get("val_loss"), (float, int))]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        selected_tensor = (r.get("selected") or {}).get("tensor")

        for cand in tested:
            is_selected = cand.get("tensor") == selected_tensor
            def cell(v):
                txt = Text(str(v))
                if is_selected:
                    txt.stylize(SELECTED_STYLE)
                return txt
            vloss = float(cand.get("val_loss", 0.0)) if isinstance(cand.get("val_loss"), (int, float)) else 0.0
            vloss_txt = heat_text(vloss, lo, hi)
            if is_selected:
                vloss_txt.stylize(SELECTED_STYLE)
            table.add_row(
                cell(cand.get("tensor", "-")),
                cell(cand.get("from_threshold", "-")),
                cell(cand.get("to_threshold", "-")),
                cell(cand.get("from_num_islands", "-")),
                cell(cand.get("to_num_islands", "-")),
                vloss_txt,
                cell(cand.get("loss_delta_pct", "-")),
                cell(cand.get("decode_token_latency_ms", "-")),
                cell(cand.get("decode_tokens_per_s", "-")),
                cell(cand.get("decode_reduction_proxy", "-")),
                cell("yes" if cand.get("feasible") else "no"),
            )


def main():
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG)
    IslandTradeoffViewer(path).run()


if __name__ == "__main__":
    main()

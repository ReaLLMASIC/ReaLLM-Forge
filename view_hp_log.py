#!/usr/bin/env python3
"""view_hp_log.py — Textual TUI that tail-views *sweep_log.yaml*

Features
========
* **Live refresh** – polls the YAML every 5 s.
* **Iterations list** – pick a number for metrics + candidate table.
* **Summary** – shows a compact table of the best config after every iteration,
  starting with the *iter -1* baseline row.

Keys
----
↑ / ↓   select in sidebar    q   quit
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
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


# ── constants ─────────────────────────────────────────────
DEFAULT_LOG = "sweep_log.yaml"
POLL_INTERVAL = 5.0  # seconds
SUMMARY_LABEL = "Summary"
HILITE_STYLE = "bold orange3"


# ── helper functions ─────────────────────────────────────
def load_yaml(path: Path) -> Dict[str, Any]:
    """Return parsed YAML or empty dict if file missing/empty."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def fnum(val: Any, spec: str) -> str:
    return spec.format(val) if isinstance(val, (int, float)) else str(val)


def human_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "-"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def hms_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "-"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_iso_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def short_time(value: Any) -> str:
    dt = parse_iso_time(value)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if dt else "-"


def elapsed_since(value: Any) -> float | None:
    dt = parse_iso_time(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()


def current_task(blk: Dict[str, Any]) -> Dict[str, Any] | None:
    for candidate in blk.get("candidates", []):
        if candidate.get("status") == "running":
            return candidate
    return None


def progress_text(candidate: Dict[str, Any] | None) -> str:
    if not candidate:
        return "-"
    done = candidate.get("completed_random_iterations", len(candidate.get("seeds", [])))
    expected = candidate.get("expected_random_iterations", "?")
    return f"{done}/{expected} seeds"


def iteration_progress(blk: Dict[str, Any]) -> str:
    completed = blk.get("completed_experiments")
    expected = blk.get("expected_experiments")
    if not isinstance(completed, int):
        completed = sum(
            int(
                candidate.get(
                    "completed_random_iterations", len(candidate.get("seeds", []))
                )
            )
            for candidate in blk.get("candidates", [])
        )
    if not isinstance(expected, int):
        candidate_expectations = [
            candidate.get("expected_random_iterations")
            for candidate in blk.get("candidates", [])
        ]
        expected = sum(v for v in candidate_expectations if isinstance(v, int))
    return f"{completed}/{expected}" if expected else f"{completed}/?"


def iteration_elapsed_seconds(blk: Dict[str, Any]) -> Any:
    if blk.get("status") == "running" and blk.get("started_at"):
        return elapsed_since(blk.get("started_at"))
    elapsed = blk.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        return elapsed
    started = parse_iso_time(blk.get("started_at"))
    updated = parse_iso_time(blk.get("last_updated_at"))
    if started and updated:
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return (
            updated.astimezone(timezone.utc) - started.astimezone(timezone.utc)
        ).total_seconds()
    return "-"


def iteration_remaining_seconds(blk: Dict[str, Any]) -> Any:
    remaining = blk.get("estimated_remaining_seconds")
    if isinstance(remaining, (int, float)):
        return remaining
    completed = blk.get("completed_experiments")
    expected = blk.get("expected_experiments")
    elapsed = iteration_elapsed_seconds(blk)
    if (
        isinstance(completed, int)
        and isinstance(expected, int)
        and completed > 0
        and expected >= completed
        and isinstance(elapsed, (int, float))
    ):
        return (elapsed / completed) * (expected - completed)
    return "-"


def eta_completion_time(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "-"
    return (datetime.now().astimezone() + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def metrics_panel(
    current_iter_baseline_metrics: Dict[str, Any],
    iteration_block: Dict[str, Any] | None = None,
    prev_iter_baseline_loss: Any = "-",
    prev_iter_baseline_rankme: Any = "-",
    prev_iter_baseline_areq: Any = "-",
    current_avg_candidate_loss: Any = "-",
    prior_avg_candidate_loss: Any = "-",
    delta_avg_candidate_loss: Any = "-",
) -> Panel:
    g = Table.grid(padding=1)
    g.add_column(justify="right")
    g.add_column()
    if iteration_block is not None:
        task = current_task(iteration_block)
        g.add_row("Status", str(iteration_block.get("status", "-")))
        g.add_row("Started", short_time(iteration_block.get("started_at")))
        if task:
            g.add_row("Running task", f"{task.get('param', '-')}={task.get('value', '-')}")
            g.add_row("Task progress", progress_text(task))
            if task.get("current_seed") is not None:
                g.add_row("Current seed", str(task.get("current_seed")))
            g.add_row("Task started", short_time(task.get("started_at")))
            elapsed = task.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) and task.get("started_at"):
                elapsed = elapsed_since(task.get("started_at"))
            g.add_row("Task elapsed", human_duration(elapsed))
            g.add_row("Task ETA", human_duration(task.get("estimated_remaining_seconds")))

    g.add_row("Prior Loss", fnum(prev_iter_baseline_loss, "{:.4f}"))
    g.add_row(
        "After Iteration Loss",
        fnum(
            current_iter_baseline_metrics.get(
                "loss", current_iter_baseline_metrics.get("best_val_loss", "-")
            ),
            "{:.4f}",
        ),
    )
    g.add_row("Best iter", str(current_iter_baseline_metrics.get("best_iter", "-")))

    g.add_row("Avg Cand Loss", fnum(current_avg_candidate_loss, "{:.4f}"))
    g.add_row("Prior Avg Loss", fnum(prior_avg_candidate_loss, "{:.4f}"))
    g.add_row("Δ Avg Loss", fnum(delta_avg_candidate_loss, "{:.4f}"))
    cur_rankme = current_iter_baseline_metrics.get("rankme", "-")
    cur_areq = current_iter_baseline_metrics.get("areq", "-")
    d_rankme = (cur_rankme - prev_iter_baseline_rankme) if isinstance(cur_rankme, (int, float)) and isinstance(prev_iter_baseline_rankme, (int, float)) else "-"
    d_areq = (cur_areq - prev_iter_baseline_areq) if isinstance(cur_areq, (int, float)) and isinstance(prev_iter_baseline_areq, (int, float)) else "-"
    g.add_row("Prior RankMe", fnum(prev_iter_baseline_rankme, "{:.4f}"))
    g.add_row("RankMe", fnum(cur_rankme, "{:.4f}"))
    g.add_row("Δ RankMe", fnum(d_rankme, "{:+.4f}"))
    g.add_row("Prior AReQ", fnum(prev_iter_baseline_areq, "{:.4f}"))
    g.add_row("AReQ", fnum(cur_areq, "{:.4f}"))
    g.add_row("Δ AReQ", fnum(d_areq, "{:+.4f}"))

    cur_rankme = current_iter_baseline_metrics.get("rankme", "-")
    cur_areq = current_iter_baseline_metrics.get("areq", "-")
    d_rankme = (
        cur_rankme - prev_iter_baseline_rankme
        if isinstance(cur_rankme, (int, float))
        and isinstance(prev_iter_baseline_rankme, (int, float))
        else "-"
    )
    d_areq = (
        cur_areq - prev_iter_baseline_areq
        if isinstance(cur_areq, (int, float)) and isinstance(prev_iter_baseline_areq, (int, float))
        else "-"
    )
    g.add_row("Prior RankMe", fnum(prev_iter_baseline_rankme, "{:.4f}"))
    g.add_row("RankMe", fnum(cur_rankme, "{:.4f}"))
    g.add_row("Δ RankMe", fnum(d_rankme, "{:+.4f}"))
    g.add_row("Prior AReQ", fnum(prev_iter_baseline_areq, "{:.4f}"))
    g.add_row("AReQ", fnum(cur_areq, "{:.4f}"))
    g.add_row("Δ AReQ", fnum(d_areq, "{:+.4f}"))

    g.add_row("Score", fnum(current_iter_baseline_metrics.get("score", "-"), "{:.4e}"))
    g.add_row(
        "Params",
        fnum(
            current_iter_baseline_metrics.get(
                "num_params", current_iter_baseline_metrics.get("params", "-")
            ),
            "{:.3e}",
        ),
    )
    g.add_row(
        "Torch alloc MB",
        fnum(
            current_iter_baseline_metrics.get(
                "peak_torch_allocated_mb",
                current_iter_baseline_metrics.get("peak_gpu_mb", "-"),
            ),
            "{:.1f}",
        ),
    )
    g.add_row(
        "Torch reserved MB",
        fnum(current_iter_baseline_metrics.get("peak_torch_reserved_mb", "-"), "{:.1f}"),
    )
    g.add_row(
        "Process GPU MB",
        fnum(current_iter_baseline_metrics.get("peak_process_gpu_mb", "-"), "{:.1f}"),
    )
    g.add_row(
        "Iter latency ms",
        fnum(current_iter_baseline_metrics.get("iter_latency_avg", "-"), "{:.2f}"),
    )

    return Panel(g, title="Iteration stats", border_style="green")


# ── TUI application ──────────────────────────────────────
class SweepViewer(App):
    CSS = """#navbox{width:16;} #main{width:1fr;}"""
    BINDINGS = [Binding("q", "quit", show=False)]

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._mtime: float = 0.0
        self.iters: List[Dict[str, Any]] = []
        self.idx: int = 0
        self.show_summary: bool = False
        self.base_cfg: Dict[str, Any] = {}
        self.base_metrics: Dict[str, Any] = {}
        self.base_iter: Dict[str, Any] | None = None

    def compose(self):  # type: ignore[override]
        yield Header()
        with Horizontal():
            with Vertical(id="navbox"):
                yield Static("Iterations", classes="title")
                yield DataTable(id="nav", show_header=False, zebra_stripes=True)
            with Vertical(id="main"):
                yield Static(id="panel")
                yield DataTable(id="table", zebra_stripes=True)
        yield Footer()

    def on_mount(self):  # type: ignore[override]
        self._load_yaml(initial=True)
        self._build_nav()
        self._refresh_view()
        self.set_interval(POLL_INTERVAL, self._poll_yaml)

    def _poll_yaml(self):
        mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0
        if mtime == self._mtime:
            return
        remember = None
        if self.iters and not self.show_summary:
            remember = self.iters[self.idx]["iter"]
        self._load_yaml()
        self._build_nav()
        if remember is not None:
            for i, it in enumerate(self.iters):
                if it["iter"] == remember:
                    self.idx = i
                    break
        self._refresh_view()

    def _load_yaml(self, *, initial: bool = False):
        data = load_yaml(self.log_path)
        self.base_cfg = data.get("baseline_config", {})
        self.base_metrics = data.get("baseline_metrics", {})
        self.base_iter = None
        for it in data.get("iterations", []):
            if it.get("iter") == -1:
                self.base_iter = it
                self.base_metrics = it.get("baseline_metrics", self.base_metrics)
                break
        self.iters = [
            it
            for it in data.get("iterations", [])
            if it.get("iter", 0) >= 0 and "candidates" in it
        ]
        if self.iters:
            self.idx = len(self.iters) - 1 if initial else min(self.idx, len(self.iters) - 1)
        else:
            self.idx = 0
        self._mtime = self.log_path.stat().st_mtime if self.log_path.exists() else 0.0

    def _build_nav(self):
        nav = self.query_one("#nav", DataTable)
        nav.clear(columns=True)
        nav.add_column("item")
        for it in self.iters:
            nav.add_row(str(it["iter"]))
        nav.add_row(SUMMARY_LABEL)
        nav.cursor_type = "row"
        row_coord = (
            self.idx if (self.iters and not self.show_summary) else len(self.iters),
            0,
        )
        nav.cursor_coordinate = row_coord
        nav.focus()

    def on_data_table_row_highlighted(self, e: DataTable.RowHighlighted):  # type: ignore[override]
        if e.data_table.id != "nav":
            return
        row = e.cursor_row  # type: ignore[attr-defined]
        self.show_summary = row == len(self.iters)
        if not self.show_summary and self.iters:
            self.idx = row
        self._refresh_view()

    def _calculate_avg_cand_loss(self, blk: Dict[str, Any]) -> float | None:
        total_loss = 0.0
        loss_count = 0
        for candidate in blk.get("candidates", []):
            loss = candidate.get("best_val_loss")
            if isinstance(loss, (int, float)):
                total_loss += loss
                loss_count += 1
        if loss_count > 0:
            return total_loss / loss_count
        return None

    def _summary_data(self):
        if not self.iters:
            return ["iter"], [["-1"]]

        changed = sorted({it["chosen"]["param"] for it in self.iters if it.get("chosen")})
        hdrs = [
            "iter",
            "elapsed",
            "progress",
            "ETA",
            "ETA done",
            *changed,
            "best_loss",
            "best_iter",
            "rankme",
            "Δrankme",
            "areq",
            "Δareq",
            "params",
            "alloc_mb",
            "resv_mb",
            "proc_mb",
            "Δparams",
            "Δalloc",
            "Δresv",
            "Δproc",
            "Δiter",
            "eff.",
        ]

        def _lookup(cfg: Dict[str, Any], key: str) -> Any:
            m = re.fullmatch(r"(\w+_layerlist)\[(\d+)\]", key)
            if not m:
                return cfg.get(key, "-")
            list_key, idx_s = m.groups()
            idx = int(idx_s)
            lst = cfg.get(list_key)
            if isinstance(lst, list) and idx < len(lst):
                return lst[idx]
            return "-"

        rows: List[List[Any]] = []

        base_src = (self.base_iter or {}).get("baseline_config_after", self.base_cfg)
        base_vals = [str(_lookup(base_src, p)) for p in changed]
        rows.append(
            [
                "-1",
                "-",
                "-",
                "-",
                "-",
                *base_vals,
                fnum(self.base_metrics.get("loss", "-"), "{:.4f}"),
                str(self.base_metrics.get("best_iter", "-")),
                fnum(self.base_metrics.get("rankme", "-"), "{:.4f}"),
                "-",
                fnum(self.base_metrics.get("areq", "-"), "{:.4f}"),
                "-",
                fnum(self.base_metrics.get("params", "-"), "{:,}"),
                fnum(
                    self.base_metrics.get(
                        "peak_torch_allocated_mb",
                        self.base_metrics.get("peak_gpu_mb", "-"),
                    ),
                    "{:.1f}",
                ),
                fnum(self.base_metrics.get("peak_torch_reserved_mb", "-"), "{:.1f}"),
                fnum(self.base_metrics.get("peak_process_gpu_mb", "-"), "{:.1f}"),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            ]
        )

        for i, it in enumerate(self.iters):
            ch = it.get("chosen") or {}
            after = it["baseline_config_after"]

            vals: List[Any] = []
            for p in changed:
                val = _lookup(after, p)
                if ch and p == ch.get("param"):
                    vals.append(Text(str(val), style=HILITE_STYLE))
                else:
                    vals.append(str(val))

            if ch:
                vals += [
                    f"{ch['best_val_loss']:.4f}",
                    str(ch.get("best_iter", "-")),
                    fnum(ch.get("avg_rankme", "-"), "{:.4f}"),
                    fnum(ch.get("delta_rankme", "-"), "{:+.4f}"),
                    fnum(ch.get("avg_areq", "-"), "{:.4f}"),
                    fnum(ch.get("delta_areq", "-"), "{:+.4f}"),
                    f"{int(ch['num_params']):,}",
                    f"{ch.get('peak_torch_allocated_mb', ch.get('peak_gpu_mb', float('nan'))):.1f}",
                    f"{ch.get('peak_torch_reserved_mb', float('nan')):.1f}",
                    f"{ch.get('peak_process_gpu_mb', float('nan')):.1f}",
                    f"{int(ch['delta_params']):,}",
                    f"{ch.get('delta_torch_allocated_mb', float('nan')):.1f}",
                    f"{ch.get('delta_torch_reserved_mb', float('nan')):.1f}",
                    f"{ch.get('delta_process_gpu_mb', float('nan')):.1f}",
                    f"{ch.get('delta_iter_latency', float('nan')):.2f}",
                    f"{ch['efficiency']:.2e}",
                ]
            else:
                baseline_metrics = it.get("baseline_metrics", {})
                vals += [
                    fnum(baseline_metrics.get("loss", "-"), "{:.4f}"),
                    str(baseline_metrics.get("best_iter", "-")),
                    fnum(baseline_metrics.get("rankme", "-"), "{:.4f}"),
                    "-",
                    fnum(baseline_metrics.get("areq", "-"), "{:.4f}"),
                    "-",
                    fnum(baseline_metrics.get("params", "-"), "{:,}"),
                    fnum(
                        baseline_metrics.get(
                            "peak_torch_allocated_mb",
                            baseline_metrics.get("peak_gpu_mb", "-"),
                        ),
                        "{:.1f}",
                    ),
                    fnum(baseline_metrics.get("peak_torch_reserved_mb", "-"), "{:.1f}"),
                    fnum(baseline_metrics.get("peak_process_gpu_mb", "-"), "{:.1f}"),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            remaining = iteration_remaining_seconds(it)
            rows.append(
                [
                    str(it.get("iter", i)),
                    hms_duration(iteration_elapsed_seconds(it)),
                    iteration_progress(it),
                    hms_duration(remaining),
                    eta_completion_time(remaining),
                    *vals,
                ]
            )
        return hdrs, rows

    def _refresh_view(self):
        panel = self.query_one("#panel", Static)
        table = self.query_one("#table", DataTable)

        if not self.iters:
            table.visible = False
            panel.update(Panel("Waiting for data… (polling)", border_style="red"))
            self.sub_title = "No data yet"
            return

        table.visible = True
        if self.show_summary:
            panel.update(Panel("Summary (best config per iteration)", border_style="cyan"))
            hdrs, rows = self._summary_data()
            table.clear(columns=True)
            table.add_columns(*hdrs)
            for r in rows:
                table.add_row(*[c if isinstance(c, Text) else str(c) for c in r])
            self.sub_title = "Summary view"
            return

        blk = self.iters[self.idx]
        current_baseline_metrics = blk["baseline_metrics"]

        prior_loss_val = "-"
        prior_rankme_val = "-"
        prior_areq_val = "-"
        if self.idx == 0:
            if self.base_metrics:
                prior_loss_val = self.base_metrics.get(
                    "loss", self.base_metrics.get("best_val_loss", "-")
                )
                prior_rankme_val = self.base_metrics.get("rankme", "-")
                prior_areq_val = self.base_metrics.get("areq", "-")
        elif self.idx > 0:
            prev_iter_baseline_metrics = self.iters[self.idx - 1]["baseline_metrics"]
            prior_loss_val = prev_iter_baseline_metrics.get(
                "loss", prev_iter_baseline_metrics.get("best_val_loss", "-")
            )
            prior_rankme_val = prev_iter_baseline_metrics.get("rankme", "-")
            prior_areq_val = prev_iter_baseline_metrics.get("areq", "-")

        current_avg_loss_val = self._calculate_avg_cand_loss(blk)
        prior_avg_loss_val = None
        if self.idx > 0:
            prior_blk = self.iters[self.idx - 1]
            prior_avg_loss_val = self._calculate_avg_cand_loss(prior_blk)

        delta_avg_loss_val = None
        if isinstance(current_avg_loss_val, float) and isinstance(prior_avg_loss_val, float):
            delta_avg_loss_val = current_avg_loss_val - prior_avg_loss_val

        current_avg_loss_display = (
            f"{current_avg_loss_val:.4f}" if current_avg_loss_val is not None else "-"
        )
        prior_avg_loss_display = (
            f"{prior_avg_loss_val:.4f}" if prior_avg_loss_val is not None else "-"
        )
        delta_avg_loss_display = (
            f"{delta_avg_loss_val:.4f}" if delta_avg_loss_val is not None else "-"
        )

        panel.update(
            metrics_panel(
                current_iter_baseline_metrics=current_baseline_metrics,
                iteration_block=blk,
                prev_iter_baseline_loss=prior_loss_val,
                current_avg_candidate_loss=current_avg_loss_display,
                prior_avg_candidate_loss=prior_avg_loss_display,
                delta_avg_candidate_loss=delta_avg_loss_display,
                prev_iter_baseline_rankme=prior_rankme_val,
                prev_iter_baseline_areq=prior_areq_val,
            )
        )
        self.sub_title = f"Iteration {blk['iter']}  (↑/↓ nav, q quit)"
        table.clear(columns=True)
        table.add_columns(
            "status",
            "progress",
            "seed",
            "started",
            "elapsed",
            "ETA",
            "param",
            "value",
            "best_loss",
            "best_iter",
            "rankme",
            "Δrankme",
            "areq",
            "Δareq",
            "alloc_mb",
            "resv_mb",
            "proc_mb",
            "Δscore",
            "Δparams",
            "Δalloc",
            "Δresv",
            "Δproc",
            "Δiter",
            "eff.",
        )

        chosen = blk.get("chosen") or {}
        for c in blk["candidates"]:
            hl = (
                bool(chosen)
                and c["param"] == chosen.get("param")
                and c["value"] == chosen.get("value")
            )
            st = "bold yellow" if hl else ""
            elapsed = c.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) and c.get("started_at"):
                elapsed = elapsed_since(c.get("started_at"))
            table.add_row(
                Text(str(c.get("status", "complete")), style=st),
                Text(progress_text(c), style=st),
                Text(str(c.get("current_seed", "-")), style=st),
                Text(short_time(c.get("started_at")), style=st),
                Text(human_duration(elapsed), style=st),
                Text(human_duration(c.get("estimated_remaining_seconds")), style=st),
                Text(str(c.get("param", "-")), style=st),
                Text(str(c.get("value", "-")), style=st),
                fnum(c.get("best_val_loss", c.get("avg_loss", "-")), "{:.4f}"),
                str(c.get("best_iter", "-")),
                fnum(c.get("avg_rankme", "-"), "{:.4f}"),
                fnum(c.get("delta_rankme", "-"), "{:+.4f}"),
                fnum(c.get("avg_areq", "-"), "{:.4f}"),
                fnum(c.get("delta_areq", "-"), "{:+.4f}"),
                fnum(c.get('peak_torch_allocated_mb', c.get('peak_gpu_mb', "-")), "{:.1f}"),
                fnum(c.get('peak_torch_reserved_mb', "-"), "{:.1f}"),
                fnum(c.get('peak_process_gpu_mb', "-"), "{:.1f}"),
                fnum(c.get('delta_score', "-"), "{:.2e}"),
                fnum(c.get('delta_params', "-"), "{:.2e}"),
                fnum(c.get('delta_torch_allocated_mb', "-"), "{:.1f}"),
                fnum(c.get('delta_torch_reserved_mb', "-"), "{:.1f}"),
                fnum(c.get('delta_process_gpu_mb', "-"), "{:.1f}"),
                fnum(c.get('delta_iter_latency', "-"), "{:.2f}"),
                fnum(c.get('efficiency', "-"), "{:.2e}"),
            )


# ── entry point ─────────────────────────────
def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_LOG)
    SweepViewer(path).run()


if __name__ == "__main__":
    main()

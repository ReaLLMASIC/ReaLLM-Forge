# plot_view.py
"""
Utility for visualising the “current view” from the Textual hyper-parameter
monitor.

Features
--------
• Scatter plot with matplotlib.
• Legend entry for every point (no text clutter on the canvas).
• Title formatted as “<y-col> vs <x-col>”.
• Optional lines connecting points that share the same *group key*.

Typical use inside the TUI
--------------------------
    import plot_view
    plot_view.plot_rows(
        current_entries,
        x="best_val_iter",
        y="best_val_loss",
        label="formatted_name",
        connect_by="seed",   # optional
    )

CLI usage
---------
    $ python plot_view.py view.json \
          --x best_val_iter \
          --y best_val_loss \
          --label formatted_name \
          --connect-by seed
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # multi-axis helper
import numpy as np
import pandas as pd
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, DefaultDict

import itertools
import os
import matplotlib.pyplot as plt
import textwrap

# ──────────────────────────── file-saving helper ───────────────────────────
PLOT_DIR = "rem_plots" #  all plots will be saved here by default
os.makedirs(PLOT_DIR, exist_ok=True)

def _safe_path(title: str, ext: str = ".png") -> str:
    """Sanitise *title* into a filename inside *PLOT_DIR*."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title)
    return os.path.join(PLOT_DIR, safe + ext)

# ───────────────────────────── title-wrap helper ───────────────────────────
def _wrap_title(title: str, width: int = 60, *, html: bool = False) -> str:
    """
    Return *title* wrapped at *width* characters.

    Matplotlib understands newlines, whereas Plotly needs <br>.
    """
    sep = "<br>" if html else "\n"
    return sep.join(textwrap.wrap(title, width=width))

# ───────────────────────────── helpers ──────────────────────────────


def _extract(entry: Dict[str, Any], key: str) -> Any:
    """Return entry[key] if present, else entry["config"][key], else None."""
    if key in entry:
        return entry[key]
    return entry.get("config", {}).get(key)


# ───────────────────────────── public API ───────────────────────────


def plot_rows(
    rows: List[Dict[str, Any]],
    *,
    x: str,
    y: str,
    label: str = "formatted_name",
    connect_by: str | None = None,
    connect_label: str | None = None,
    fit_line: bool = False,
) -> None:
    """
    Scatter-plot *y* vs *x*.

    Parameters
    ----------
    rows : list of run dictionaries.
    x, y : str
        Column names for the X and Y axes.
    label : str, default "formatted_name"
        Column used for legend labels (duplicates collapsed).
    connect_by : str | None
        If provided, points that share the same value for this key are joined
        by a line (sorted by X).
    """
    xs, ys, lbls = [], [], []
    groups: DefaultDict[Any, List[tuple[float, float, str]]] = defaultdict(list)

    for run in rows:
        xv, yv = _extract(run, x), _extract(run, y)
        if xv is None or yv is None:
            continue
        lbl = str(_extract(run, label) or "")
        if connect_by:
            gid = _extract(run, connect_by)
            groups[gid].append((xv, yv, lbl))
        else:
            xs.append(xv)
            ys.append(yv)
            lbls.append(lbl)

    if connect_by:
        # Flatten groups for scatter plotting
        for pts in groups.values():
            for xv, yv, lbl in pts:
                xs.append(xv)
                ys.append(yv)
                lbls.append(lbl)

    if not xs:
        raise ValueError("No plottable data in supplied rows")

    fig = plt.figure(figsize=(12, 8))

    seen: Set[str] = set()
    if not connect_by:
        for xv, yv, lbl in zip(xs, ys, lbls):
            kw = {"label": lbl} if lbl not in seen else {"label": "_nolegend_"}
            seen.add(lbl)
            plt.scatter(xv, yv, s=40, **kw)

    if connect_by:

        # Consistent colours for points & their connecting line
        colour_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        for gid, pts in groups.items():
            col = next(colour_cycle)

            # Plot individual points in this group
            for xv, yv, _ in pts:
                plt.scatter(xv, yv, s=40, color=col, label="_nolegend_")

            # Plot connecting line (if ≥2 points)
            if len(pts) >= 2:
                pts_sorted = sorted(pts, key=lambda t: t[0])  # sort by X
                plt.plot(
                    [p[0] for p in pts_sorted],
                    [p[1] for p in pts_sorted],
                    color=col,
                    linewidth=1,
                    alpha=0.7,
                    label=f"{(connect_label or connect_by)}={gid}",   # NEW
                )

    if fit_line and xs:
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        y_pred = slope * x_arr + intercept
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        order = np.argsort(x_arr)
        plt.plot(x_arr[order], y_pred[order], color="black", linestyle="--", linewidth=1)
        eq_text = f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r2:.4f}"
        plt.text(
            0.02,
            0.98,
            eq_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.xlabel(x)
    plt.ylabel(y)
    title = f"{y} vs {x}"
    if connect_by:
        disp = connect_label or connect_by   # NEW
        title += f"  (lines grouped by '{disp}')"
    plt.title(_wrap_title(title))
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig(_safe_path(title), dpi=300)
    plt.show()


# ───────────────────────────── plot bars ──────────────────────────────
def _extract(entry: Dict[str, Any], key: str) -> Any:
    """Return entry[key] if present, else entry['config'][key], else None."""
    if key in entry:
        return entry[key]
    return entry.get("config", {}).get(key)



def plot_bars(
    rows: List[Dict[str, Any]],
    *,
    y: str,
    label_cols: List[str],
    bar_color: str = "#1f77b4",
) -> None:
    if not label_cols:
        raise ValueError("Need ≥1 label column for bar chart")

    labels, heights = [], []
    for r in rows:
        yv = _extract(r, y)
        if yv is None:
            continue
        merged = "-".join(
            "None" if _extract(r, c) is None else str(_extract(r, c))
            for c in label_cols
        )
        labels.append(merged)
        heights.append(float(yv))

    if not heights:
        raise ValueError(f"No numeric values found in column “{y}”")

    order = np.argsort(np.array(heights))
    df = pd.DataFrame(
        {"label": np.array(labels)[order], y: np.array(heights)[order]}
    )

    # pretty title parts
    pretty = lambda s: s.replace("_", " ").title()
    title_text = f"{pretty(y)} by {' / '.join(pretty(c) for c in label_cols)}"

    fig = px.bar(
        df,
        x=y,
        y="label",
        orientation="h",
        color_discrete_sequence=[bar_color],
        text=df[y].apply(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
            title=dict(
                text=_wrap_title(title_text, html=True),
                x=0.5,             # halfway across the plot *area*
                xref="container",  # ← key line: ignore the margins
                xanchor="center",
                yanchor="top",
                ),
            # merged column names become the Y-axis title
            yaxis=dict(
                title=" / ".join(label_cols),   # e.g. "optimizer / sgd_nesterov"
                autorange="reversed",           # keep smallest bar on top
                ),
            xaxis=dict(title=y),
            margin=dict(l=120, r=40, t=80, b=40),  # t a bit larger for the centred title
    )

    fig.write_image(_safe_path(title_text), scale=2)  # requires kaleido
    fig.show()

# ───────────────────────────── plot multi bars ──────────────────────────────
def plot_multi_bars(
    rows: List[Dict[str, Any]],
    *,
    y_cols: List[str],          # numeric columns to plot (≥1)
    label_cols: List[str],      # columns whose values build the category label (≥1)
) -> None:
    if not y_cols or not label_cols:
        raise ValueError("Need ≥1 numeric-col *and* ≥1 label-col")

    # ---- harvest data -----------------------------------------------------
    cats: List[str] = []
    series: Dict[str, List[float]] = {yc: [] for yc in y_cols}

    for r in rows:
        merged = "-".join(
            "None" if _extract(r, c) is None else str(_extract(r, c))
            for c in label_cols
        )
        cat_label = merged
        if cat_label not in cats:
            cats.append(cat_label)
        for yc in y_cols:
            val = _extract(r, yc)
            series[yc].append(float(val) if val is not None else np.nan)

    # ---- build traces -----------------------------------------------------
    # ── choose layout: one secondary axis (2 metrics) or stacked rows (>2) ──
    if len(y_cols) == 1:
        # ----- single metric → simple grouped bars ----------------------
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cats,
                y=series[y_cols[0]],
                name=y_cols[0],
                marker_color=px.colors.qualitative.Plotly[0],
            )
        )
        fig.update_layout(
            barmode="group",
            xaxis_title=" / ".join(label_cols),
            yaxis_title=y_cols[0],
        )
    else:
        # N > 2  → one row per metric, shared X axis
        fig = make_subplots(
            rows=len(y_cols),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
        )
        col_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        for idx, yc in enumerate(y_cols, start=1):
            fig.add_trace(
                go.Bar(
                    x=cats,
                    y=series[yc],
                    name=yc,
                    marker_color=next(col_cycle),
                    showlegend=False,
                ),
                row=idx,
                col=1,
            )
        # give each panel its own y-axis label
            fig.update_yaxes(title_text=yc, row=idx, col=1)
        fig.update_layout(barmode="group", height=300 * len(y_cols))

    pretty = lambda s: s.replace("_", " ").title()
    # ── common layout bits ────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=_wrap_title(
                 f"{' / '.join(map(pretty, y_cols))} by "
                 f"{' / '.join(map(pretty, label_cols))}",
                html=True,
             ),
            x=0.5, xanchor="center", yanchor="top",
        ),
        bargap=0.15,
        bargroupgap=0.1,
        legend_title_text="Metric",
    )

    fig.write_image(_safe_path(fig.layout.title.text), scale=2)  # kaleido
    fig.show()


# ─────────────── multi-metric trimmed bar-chart (Δ view) ────────────────
def plot_multi_bars_trim(                     # NEW
    rows: List[Dict[str, Any]],
    *,
    y_cols: List[str],
    label_cols: List[str],
) -> None:
    """
    Same signature as ``plot_multi_bars`` but each metric is shown after
    subtracting its own minimum, letting you inspect the spread.
    """
    if not y_cols or not label_cols:
        raise ValueError("Need ≥1 numeric-col and ≥1 label-col")

    cats: List[str] = []
    raw: Dict[str, List[float]] = {c: [] for c in y_cols}

    for r in rows:
        merged = "-".join(
            "None" if _extract(r, c) is None else str(_extract(r, c))
            for c in label_cols
        )
        if merged not in cats:
            cats.append(merged)
        for yc in y_cols:
            v = _extract(r, yc)
            raw[yc].append(float(v) if v is not None else np.nan)

    # trim common baseline per metric
    diffs = {
        yc: [v - np.nanmin(raw[yc]) for v in raw[yc]]
        for yc in y_cols
    }

    if len(y_cols) == 1:
        yc = y_cols[0]
        fig = px.bar(
            x=cats, y=diffs[yc], labels={"x": " / ".join(label_cols), "y": yc}
        )
    else:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(
            rows=len(y_cols), cols=1, shared_xaxes=True, vertical_spacing=0.03
        )
        palette = itertools.cycle(px.colors.qualitative.Plotly)
        for i, yc in enumerate(y_cols, 1):
            fig.add_trace(
                go.Bar(x=cats, y=diffs[yc], marker_color=next(palette), showlegend=False),
                row=i, col=1,
            )
            fig.update_yaxes(title_text=f"{yc} Δ", row=i, col=1)
        fig.update_layout(height=300 * len(y_cols))

    pretty = lambda s: s.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"Δ({' / '.join(map(pretty, y_cols))}) by "
                 f"{' / '.join(map(pretty, label_cols))}",
            x=0.5, xanchor="center", yanchor="top",
        ),
        bargap=0.15, bargroupgap=0.1,
    )

    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in fig.layout.title.text)
    fig.write_image(f"{safe}.png", scale=2)
    fig.show()

def plot_bars_trim(
    rows: List[Dict[str, Any]],
    *,
    y: str,
    label_cols: List[str],
) -> None:
    """
    Backward-compatible single-metric Δ-bar helper used by monitor ``z`` hotkey.

    This intentionally mirrors ``plot_bars`` args while delegating to the
    multi-metric trimmed implementation.
    """
    if not label_cols:
        raise ValueError("Need ≥1 label column for bar chart")
    plot_multi_bars_trim(rows, y_cols=[y], label_cols=label_cols)


# ───────────────────────────── CLI wrapper ──────────────────────────


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Scatter-plot an exported monitor view (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("view_json", type=Path, help="Path to JSON from the TUI")
    ap.add_argument("--x", default="best_val_iter", help="Key for the X axis")
    ap.add_argument("--y", default="best_val_loss", help="Key for the Y axis")
    ap.add_argument("--label", default="formatted_name", help="Key for legend labels")
    ap.add_argument(
        "--connect-by",
        dest="connect_by",
        default=None,
        help="Join points that share this key with a line",
    )
    args = ap.parse_args()

    data = json.loads(args.view_json.read_text())["rows"]
    plot_rows(
        data,
        x=args.x,
        y=args.y,
        label=args.label,
        connect_by=args.connect_by,
    )


if __name__ == "__main__":
    _cli()

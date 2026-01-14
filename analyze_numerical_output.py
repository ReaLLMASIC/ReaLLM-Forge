#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HEADER_RE = re.compile(r'^\s*sinewave/s(\d+)\s*:\s*$', re.IGNORECASE)
INT_RE = re.compile(r'-?\d+')  # grabs numbers and ignores stray tokens

def parse_blocks(text: str) -> Dict[str, List[int]]:
    lines = text.splitlines()
    data: Dict[str, List[int]] = {}
    current_key: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_key
        if current_key is None:
            return
        joined = "\n".join(buffer)
        nums = [int(x) for x in INT_RE.findall(joined)]
        data[current_key] = nums
        buffer = []

    for line in lines:
        m = HEADER_RE.match(line.strip())
        if m:
            flush()
            base_key = f"s{int(m.group(1))}"
            key = base_key
            # If header repeats later, keep _2, _3, etc.
            i = 2
            while key in data:
                key = f"{base_key}_{i}"
                i += 1
            current_key = key
            buffer = []
        else:
            if current_key is not None:
                buffer.append(line)
    flush()
    return {k: v for k, v in data.items() if v}

def load_series(path: Path) -> Dict[str, List[int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    series = parse_blocks(text)
    if not series:
        raise ValueError("No series found. Ensure lines like 'sinewave/s1:' precede numeric data.")
    return series

def maybe_slice(values: List[int], start: Optional[int], end: Optional[int]) -> List[int]:
    if start is None and end is None:
        return values
    n = len(values)
    s = 0 if start is None else max(0, start)
    e = n if end is None else min(n, end)
    return values[s:e]

def sort_key(name: str):
    base, _, dup = name.partition("_")
    try:
        num = int(base[1:])
    except ValueError:
        num = 10**9
    dup_num = int(dup) if dup.isdigit() else 0
    return (num, dup_num, name)

def make_figure(
    series: Dict[str, List[int]],
    title: str,
    x_label: str,
    y_label: str,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    start: Optional[int],
    end: Optional[int],
    height: int,
    show_markers: bool,
) -> go.Figure:

    fig = make_subplots(rows=1, cols=1)

    for name in sorted(series.keys(), key=sort_key):
        y = maybe_slice(series[name], start, end)
        x = list(range(len(y)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers" if show_markers else "lines",
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    + x_label + ": %{x}<br>"
                    + y_label + ": %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
            title=title,
            height=height,
            template="plotly",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=60, r=20, t=60, b=60),
            xaxis=dict(
                title=x_label,
                rangeslider=dict(visible=True),
                # No rangeselector buttons here—those are date-based only.
                ),
            yaxis=dict(title=y_label),
            )


    # Respect optional axis limits
    if xlim:
        fig.update_xaxes(range=[xlim[0], xlim[1]])
    if ylim:
        fig.update_yaxes(range=[ylim[0], ylim[1]])

    # Patch rangeselector steps to act like index-count shortcuts
    # (Plotly doesn’t have index-based steps; we emulate via updatemenus)
    # We add a small updatemenu to jump to last N points by setting xaxis range.
    # If there are multiple traces of different lengths, use max length.
    max_len = max((len(v) for v in series.values()), default=0)
    def end_range(n):
        r = max_len
        l = max(0, r - n)
        return [l, r]

    fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    y=1.12,
                    xanchor="left",
                    yanchor="bottom",
                    showactive=False,
                    buttons=[
                        dict(label="Last 100", method="relayout", args=[{"xaxis.range": end_range(100)}]),
                        dict(label="Last 500", method="relayout", args=[{"xaxis.range": end_range(500)}]),
                        dict(label="Last 1k", method="relayout", args=[{"xaxis.range": end_range(1000)}]),
                        dict(label="All", method="relayout", args=[{"xaxis.autorange": True}]),
                        ],
                    )
                ]
            )


    return fig

def main():
    p = argparse.ArgumentParser(
        description="Plot sinewave series from a text file with 'sinewave/s#:' headers using Plotly (interactive)."
    )
    p.add_argument("input", type=Path, help="Path to the input text file.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="If set to a .html file, saves an interactive HTML. "
                        "If set to .png/.pdf/.svg requires 'kaleido' installed.")
    p.add_argument("--title", default="Sinewave Series (Plotly)")
    p.add_argument("--xlabel", default="Index")
    p.add_argument("--ylabel", default="Value")
    p.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"))
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"))
    p.add_argument("--start", type=int, default=None, help="Slice start index (inclusive) for each series.")
    p.add_argument("--end", type=int, default=None, help="Slice end index (exclusive) for each series.")
    p.add_argument("--height", type=int, default=720, help="Figure height in pixels.")
    p.add_argument("--show-markers", action="store_true", help="Draw small markers along the lines.")
    p.add_argument("--no-open", action="store_true", help="Don’t open a browser window; just save output if requested.")
    args = p.parse_args()

    series = load_series(args.input)
    fig = make_figure(
        series=series,
        title=args.title,
        x_label=args.xlabel,
        y_label=args.ylabel,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        start=args.start,
        end=args.end,
        height=args.height,
        show_markers=args.show_markers,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".html":
            fig.write_html(str(args.output), include_plotlyjs="cdn", full_html=True)
            print(f"Saved interactive HTML to: {args.output}")
        else:
            # For static export, you need: pip install -U kaleido
            fig.write_image(str(args.output))
            print(f"Saved static image to: {args.output}")

    if not args.no_open and not args.output:
        # Open in a temporary browser tab (works in most environments)
        fig.show()

if __name__ == "__main__":
    main()


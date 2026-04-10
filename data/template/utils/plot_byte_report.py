#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go


def _parse_simple_yaml(path: Path) -> Dict[str, float]:
    data: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.isdigit():
            data[key] = int(value)
        else:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def _load_report(path: Path) -> Dict[str, float]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        return loaded
    raise ValueError(f"Unexpected YAML format in {path}")


def _label_from_path(path: Path, mode: str) -> str:
    if mode == "filename":
        return path.stem
    if mode == "parent":
        return path.parent.name or path.stem
    stem = path.stem
    if stem.startswith("byte_token_report"):
        return path.parent.name or stem
    return stem


def _build_counts_figure(labels: List[str], byte_counts: List[int], non_byte_counts: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="byte_tokens", x=labels, y=byte_counts)
    fig.add_bar(name="non_byte_tokens", x=labels, y=non_byte_counts)
    fig.update_layout(
        barmode="group",
        title="Byte vs Non-Byte Token Counts",
        xaxis_title="Tokenizer Run",
        yaxis_title="Token Count",
    )
    return fig


def _build_percent_figure(labels: List[str], byte_pct: List[float], non_byte_pct: List[float], mode: str) -> go.Figure:
    if mode == "stacked":
        fig = go.Figure()
        fig.add_bar(name="byte_percentage", x=labels, y=byte_pct)
        fig.add_bar(name="non_byte_percentage", x=labels, y=non_byte_pct)
        fig.update_layout(
            barmode="stack",
            title="Byte vs Non-Byte Token Percentages",
            xaxis_title="Tokenizer Run",
            yaxis_title="Percentage",
            yaxis=dict(range=[0, 100]),
        )
        return fig
    if mode == "line":
        fig = go.Figure()
        fig.add_scatter(name="byte_percentage", x=labels, y=byte_pct, mode="lines+markers")
        fig.add_scatter(name="non_byte_percentage", x=labels, y=non_byte_pct, mode="lines+markers")
        fig.update_layout(
            title="Byte vs Non-Byte Token Percentages",
            xaxis_title="Tokenizer Run",
            yaxis_title="Percentage",
            yaxis=dict(range=[0, 100]),
        )
        return fig
    if mode == "pie":
        fig = go.Figure()
        fig.add_trace(go.Pie(labels=["byte", "non-byte"], values=[sum(byte_pct), sum(non_byte_pct)]))
        fig.update_layout(title="Average Byte vs Non-Byte Percentage (Pie)")
        return fig
    fig = go.Figure()
    fig.add_bar(name="byte_percentage", x=labels, y=byte_pct)
    fig.add_bar(name="non_byte_percentage", x=labels, y=non_byte_pct)
    fig.update_layout(
        barmode="group",
        title="Byte vs Non-Byte Token Percentages",
        xaxis_title="Tokenizer Run",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100]),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot byte token reports from YAML files into PNGs using Plotly."
    )
    parser.add_argument("reports", nargs="+", help="YAML report files to compare.")
    parser.add_argument(
        "--output-dir",
        default="./byte_report",
        help="Directory where PNG/HTML outputs will be written.",
    )
    parser.add_argument(
        "--percent-plot",
        choices=["grouped", "stacked", "line", "pie"],
        default="grouped",
        help="Chart style for percentage comparison.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["auto", "filename", "parent"],
        default="auto",
        help="How to derive labels for each report.",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Write HTML versions of the charts for interactive comparison.",
    )
    args = parser.parse_args()

    report_paths = [Path(path) for path in args.reports]
    labels: List[str] = []
    byte_counts: List[int] = []
    non_byte_counts: List[int] = []
    byte_pct: List[float] = []
    non_byte_pct: List[float] = []

    for report_path in report_paths:
        report = _load_report(report_path)
        labels.append(_label_from_path(report_path, args.label_mode))
        byte_counts.append(int(report["byte_tokens"]))
        non_byte_counts.append(int(report["non_byte_tokens"]))
        byte_pct.append(float(report["byte_percentage"]))
        non_byte_pct.append(float(report["non_byte_percentage"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_fig = _build_counts_figure(labels, byte_counts, non_byte_counts)
    counts_png = output_dir / "byte_token_counts.png"
    counts_fig.write_image(str(counts_png))

    percent_fig = _build_percent_figure(labels, byte_pct, non_byte_pct, args.percent_plot)
    percent_png = output_dir / "byte_token_percentages.png"
    percent_fig.write_image(str(percent_png))

    if args.write_html:
        counts_fig.write_html(str(output_dir / "byte_token_counts.html"))
        percent_fig.write_html(
            str(output_dir / f"byte_token_percentages_{args.percent_plot}.html")
        )
        _build_percent_figure(labels, byte_pct, non_byte_pct, "grouped").write_html(
            str(output_dir / "byte_token_percentages_grouped.html")
        )
        _build_percent_figure(labels, byte_pct, non_byte_pct, "stacked").write_html(
            str(output_dir / "byte_token_percentages_stacked.html")
        )


if __name__ == "__main__":
    main()

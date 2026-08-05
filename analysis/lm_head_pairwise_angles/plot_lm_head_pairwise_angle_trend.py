#!/usr/bin/env python3
"""Plot LM-head pairwise angle-difference metrics across checkpoint iterations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

from compare_lm_head_pairwise_angles import compare_lm_head_pairwise_angles

TREND_METRICS = [
    "diff_deg_mean",
    "diff_deg_median",
    "diff_deg_std",
    "rmse_deg",
    "avg_abs_degrees_difference",
    "median_abs_degrees_difference",
    "stddev_abs_degrees_difference",
]


def checkpoint_for_iteration(run_dir: str, iteration: int) -> str:
    return str(Path(run_dir) / f"{iteration}.pt")


def build_trend(
    run_a: str,
    run_b: str,
    iterations: List[int],
    *,
    meta: str | None,
    device: str,
    min_angle: float,
    max_angle: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for iteration in iterations:
        ckpt_a = checkpoint_for_iteration(run_a, iteration)
        ckpt_b = checkpoint_for_iteration(run_b, iteration)
        if not os.path.exists(ckpt_a) or not os.path.exists(ckpt_b):
            missing = ckpt_a if not os.path.exists(ckpt_a) else ckpt_b
            raise FileNotFoundError(f"Missing checkpoint for iteration {iteration}: {missing}")
        result = compare_lm_head_pairwise_angles(
            ckpt_a,
            ckpt_b,
            meta=meta,
            device=device,
            min_angle=min_angle,
            max_angle=max_angle,
        )
        row: Dict[str, float] = {"iteration": float(iteration)}
        row.update(result.metrics)
        rows.append(row)
    return rows


def write_trend_csv(rows: List[Dict[str, float]], csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    fieldnames = ["iteration"] + [key for key in rows[0] if key != "iteration"] if rows else ["iteration"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trend_html(rows: List[Dict[str, float]]) -> str:
    import plotly.graph_objects as go

    iterations = [row["iteration"] for row in rows]
    fig = go.Figure()
    for metric in TREND_METRICS:
        y = [row.get(metric) for row in rows]
        if metric.startswith("diff_deg") or metric == "rmse_deg":
            error = [row.get("diff_deg_std", 0.0) for row in rows]
        else:
            error = [row.get("stddev_abs_degrees_difference", 0.0) for row in rows]
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=y,
                mode="lines+markers",
                name=metric,
                error_y={"type": "data", "array": error, "visible": True},
            )
        )
    fig.update_layout(
        title="LM-head pairwise angle-difference trend",
        xaxis_title="Training iteration",
        yaxis_title="Degrees",
        hovermode="x unified",
    )
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def write_trend_html(rows: List[Dict[str, float]], html_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(html_path)) or ".", exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(trend_html(rows))


def parse_iterations(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="Directory containing iteration checkpoints for run A")
    parser.add_argument("run_b", help="Directory containing iteration checkpoints for run B")
    parser.add_argument("--iterations", default="0,200,400,600,800")
    parser.add_argument("--meta", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--min-angle", type=float, default=0.0)
    parser.add_argument("--max-angle", type=float, default=180.0)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--html", required=True)
    args = parser.parse_args()

    rows = build_trend(
        args.run_a,
        args.run_b,
        parse_iterations(args.iterations),
        meta=args.meta,
        device=args.device,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
    )
    write_trend_csv(rows, args.csv)
    write_trend_html(rows, args.html)
    print(f"wrote trend CSV: {args.csv}")
    print(f"wrote trend HTML: {args.html}")


if __name__ == "__main__":
    main()

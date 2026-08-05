#!/usr/bin/env python3
"""Monitor a CSV stream, sample a cone of multicontext continuations, and plot it.

The script waits until complete rows are appended to a target CSV file. On each
change it writes a stable prompt snapshot, runs sample.py with
--multicontext_csv_input and --num_samples <cone_width>, then updates a Plotly
HTML graph that overlays observed rows and all sampled futures per column.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def read_complete_text(path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    text = data.decode("utf-8")
    if not text.endswith(("\n", "\r")):
        last_newline = max(text.rfind("\n"), text.rfind("\r"))
        text = "" if last_newline < 0 else text[: last_newline + 1]
    return text


def count_data_rows(csv_text: str, has_header: bool) -> int:
    rows = [row for row in csv.reader(csv_text.splitlines()) if row]
    if has_header and rows:
        rows = rows[1:]
    return len(rows)


def load_manifest_datasets(manifest_path: Path) -> List[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("multicontext_datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"manifest has no multicontext_datasets: {manifest_path}")
    return [str(dataset) for dataset in datasets]


def read_numeric_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(cell) for cell in row] for row in reader if row]
    return header, rows


def write_cone_plot(prompt_csv: Path, sample_csvs: Iterable[Path], output_html: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("plotly is required for cone plotting; install with `pip install plotly`.") from exc

    headers, observed = read_numeric_csv(prompt_csv)
    sample_paths = list(sample_csvs)
    fig = make_subplots(
        rows=len(headers),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=headers,
    )

    observed_x = list(range(len(observed)))
    for col_idx, header in enumerate(headers, start=1):
        observed_y = [row[col_idx - 1] for row in observed]
        fig.add_trace(
            go.Scatter(
                x=observed_x,
                y=observed_y,
                mode="lines+markers",
                name=f"observed:{header}",
                legendgroup="observed",
                line=dict(width=3),
                showlegend=(col_idx == 1),
            ),
            row=col_idx,
            col=1,
        )

        for sample_idx, sample_path in enumerate(sample_paths, start=1):
            _, sample_rows = read_numeric_csv(sample_path)
            y_vals = [row[col_idx - 1] for row in sample_rows]
            x_vals = list(range(len(y_vals)))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=f"cone_{sample_idx}",
                    legendgroup=f"cone_{sample_idx}",
                    opacity=0.45,
                    showlegend=(col_idx == 1),
                ),
                row=col_idx,
                col=1,
            )
        fig.update_yaxes(title_text=header, row=col_idx, col=1)

    fig.update_xaxes(title_text="row index", row=len(headers), col=1)
    fig.update_layout(
        title=f"CSV multicontext cone updated {datetime.now().isoformat(timespec='seconds')}",
        template="plotly_white",
        height=max(350 * len(headers), 500),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor CSV rows and update multicontext prediction cone plot.")
    parser.add_argument("--target_csv", required=True, help="CSV file to monitor for complete appended rows.")
    parser.add_argument("--manifest", default="data/csv_mc_int/manifest.json", help="Prepared dataset manifest.json.")
    parser.add_argument("--out_dir", default="out/csv_mc_int", help="Checkpoint directory passed to sample.py.")
    parser.add_argument("--work_dir", default="out/csv_mc_int/cone_monitor", help="Snapshots, samples, and plot output directory.")
    parser.add_argument("--cone_width", type=int, default=1, help="Number of sampled futures per update.")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Rows to generate beyond the prompt.")
    parser.add_argument("--poll_seconds", type=float, default=10.0, help="Polling interval.")
    parser.add_argument("--iterations", type=int, default=0, help="Maximum updates; 0 means run forever.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no_header", dest="has_header", action="store_false", help="Target CSV has no header row.")
    args = parser.parse_args()

    target_csv = Path(args.target_csv)
    manifest_path = Path(args.manifest)
    work_dir = Path(args.work_dir)
    snapshots_dir = work_dir / "snapshots"
    samples_dir = work_dir / "samples"
    plot_path = work_dir / "cone.html"
    datasets = load_manifest_datasets(manifest_path)

    last_rows = -1
    updates = 0
    print(f"Monitoring {target_csv} for complete rows. Plot: {plot_path}")
    while True:
        complete_text = read_complete_text(target_csv)
        row_count = count_data_rows(complete_text, args.has_header)
        if row_count > last_rows and row_count > 0:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot = snapshots_dir / f"prompt_{stamp}.csv"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(complete_text, encoding="utf-8")

            run_samples_dir = samples_dir / stamp
            cmd = [
                sys.executable,
                "sample.py",
                "--out_dir",
                args.out_dir,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--compile",
                "--multicontext",
                "--multicontext_datasets",
                *datasets,
                "--multicontext_csv_input",
                str(snapshot),
                "--multicontext_csv_output_dir",
                str(run_samples_dir),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--top_k",
                "1",
                "--num_samples",
                str(args.cone_width),
                "--no-print_model_info",
            ]
            if not args.has_header:
                cmd.append("--no-multicontext_csv_has_header")

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            sample_csvs = sorted(run_samples_dir.glob("*.csv"))
            write_cone_plot(snapshot, sample_csvs, plot_path)
            print(f"Updated {plot_path} with {len(sample_csvs)} cone samples from {row_count} observed rows.")

            last_rows = row_count
            updates += 1
            if args.iterations and updates >= args.iterations:
                break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

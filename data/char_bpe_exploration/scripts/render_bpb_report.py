#!/usr/bin/env python3
"""Render a Plotly HTML report for FLORES char-BPE validation/BPB runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TRAINING_SUMMARY = Path("out/char_bpe_flores_validation_bpb/summary.csv")
DEFAULT_TOKENIZATION_SUMMARY = Path("data/char_bpe_exploration/results/summary.csv")
DEFAULT_OUTPUT = Path("out/char_bpe_flores_validation_bpb/report.html")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing CSV file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def sorted_vocab_sizes(rows: list[dict[str, str]]) -> list[int]:
    return sorted({as_int(row, "vocab_size") for row in rows if row.get("vocab_size")})


def sorted_languages(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["language"] for row in rows if row.get("language")})


def line_traces(rows: list[dict[str, str]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    traces = []
    for language in sorted(grouped):
        lang_rows = sorted(grouped[language], key=lambda row: as_int(row, "vocab_size"))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": language,
                "x": [as_int(row, "vocab_size") for row in lang_rows],
                "y": [finite_or_none(as_float(row, metric)) for row in lang_rows],
                "hovertemplate": f"{language}<br>vocab=%{{x}}<br>{metric}=%{{y:.5f}}<extra></extra>",
            }
        )
    return traces


def heatmap_trace(rows: list[dict[str, str]], metric: str) -> list[dict[str, Any]]:
    vocabs = sorted_vocab_sizes(rows)
    languages = sorted_languages(rows)
    lookup = {(row["language"], as_int(row, "vocab_size")): as_float(row, metric) for row in rows}
    z = [[finite_or_none(lookup.get((language, vocab), float("nan"))) for vocab in vocabs] for language in languages]
    return [
        {
            "type": "heatmap",
            "x": vocabs,
            "y": languages,
            "z": z,
            "colorscale": "Viridis",
            "hovertemplate": "language=%{y}<br>vocab=%{x}<br>value=%{z:.5f}<extra></extra>",
        }
    ]


def best_bar_trace(rows: list[dict[str, str]], metric: str) -> list[dict[str, Any]]:
    best_rows = []
    for language in sorted_languages(rows):
        candidates = [row for row in rows if row.get("language") == language and math.isfinite(as_float(row, metric))]
        if candidates:
            best_rows.append(min(candidates, key=lambda row: as_float(row, metric)))
    best_rows.sort(key=lambda row: as_float(row, metric))
    return [
        {
            "type": "bar",
            "orientation": "h",
            "x": [as_float(row, metric) for row in best_rows],
            "y": [row["language"] for row in best_rows],
            "customdata": [as_int(row, "vocab_size") for row in best_rows],
            "hovertemplate": "language=%{y}<br>best " + metric + "=%{x:.5f}<br>vocab=%{customdata}<extra></extra>",
        }
    ]


def scatter_loss_vs_bpb(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "scatter",
            "mode": "markers",
            "x": [as_float(row, "validation_loss") for row in rows],
            "y": [as_float(row, "bits_per_byte") for row in rows],
            "text": [row["language"] for row in rows],
            "customdata": [as_int(row, "vocab_size") for row in rows],
            "marker": {
                "size": 10,
                "color": [as_int(row, "vocab_size") for row in rows],
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "vocab"},
            },
            "hovertemplate": "language=%{text}<br>vocab=%{customdata}<br>val_loss=%{x:.5f}<br>BPB=%{y:.5f}<extra></extra>",
        }
    ]


def tokenization_lookup(tokenization_rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in tokenization_rows:
        language = row.get("language")
        vocab = row.get("requested_vocab_size") or row.get("vocab_size")
        if language and vocab:
            lookup[(language, int(float(vocab)))] = row
    return lookup


def enrich_training_rows(
    training_rows: list[dict[str, str]],
    tokenization_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    lookup = tokenization_lookup(tokenization_rows)
    enriched = []
    for row in training_rows:
        merged = dict(row)
        token_row = lookup.get((row.get("language", ""), as_int(row, "vocab_size")), {})
        for key in ("val_bytes_per_token", "val_chars_per_token", "val_tokens", "unk_byte_fallback_tokens", "actual_vocab_size"):
            if key in token_row:
                merged[key] = token_row[key]
        enriched.append(merged)
    return enriched


def metric_scatter(rows: list[dict[str, str]], x_metric: str, y_metric: str) -> list[dict[str, Any]]:
    usable = [row for row in rows if row.get(x_metric) not in (None, "") and row.get(y_metric) not in (None, "")]
    return [
        {
            "type": "scatter",
            "mode": "markers",
            "x": [as_float(row, x_metric) for row in usable],
            "y": [as_float(row, y_metric) for row in usable],
            "text": [row["language"] for row in usable],
            "customdata": [as_int(row, "vocab_size") for row in usable],
            "marker": {"size": 10, "color": [as_float(row, "bits_per_byte") for row in usable], "colorscale": "Viridis", "showscale": True},
            "hovertemplate": "language=%{text}<br>vocab=%{customdata}<br>" + x_metric + "=%{x:.5f}<br>" + y_metric + "=%{y:.5f}<extra></extra>",
        }
    ]


def make_plots(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "validation-loss-lines",
            "title": "Validation loss by char-BPE vocabulary size",
            "description": "Lower is better. Use this to see how each language responds to larger char-BPE vocabularies.",
            "data": line_traces(rows, "validation_loss"),
            "layout": {"xaxis": {"title": "Requested vocabulary size"}, "yaxis": {"title": "Validation loss (nats/token)"}},
        },
        {
            "id": "bpb-lines",
            "title": "Bits per byte by char-BPE vocabulary size",
            "description": "Lower is better. BPB normalizes validation loss by UTF-8 bytes per validation token.",
            "data": line_traces(rows, "bits_per_byte"),
            "layout": {"xaxis": {"title": "Requested vocabulary size"}, "yaxis": {"title": "Bits per byte"}},
        },
        {
            "id": "validation-loss-heatmap",
            "title": "Validation-loss heatmap",
            "description": "A compact view of all language/vocab combinations; darker/lighter changes expose language-specific scaling trends.",
            "data": heatmap_trace(rows, "validation_loss"),
            "layout": {"xaxis": {"title": "Requested vocabulary size"}, "yaxis": {"title": "Language"}},
        },
        {
            "id": "bpb-heatmap",
            "title": "Bits-per-byte heatmap",
            "description": "Highlights which vocabulary sizes give the best compression-normalized validation behavior per language.",
            "data": heatmap_trace(rows, "bits_per_byte"),
            "layout": {"xaxis": {"title": "Requested vocabulary size"}, "yaxis": {"title": "Language"}},
        },
        {
            "id": "best-bpb-bar",
            "title": "Best observed BPB per language",
            "description": "Ranks languages by their best observed BPB and shows which vocab size produced it.",
            "data": best_bar_trace(rows, "bits_per_byte"),
            "layout": {"xaxis": {"title": "Best bits per byte"}, "yaxis": {"title": "Language", "automargin": True}},
        },
        {
            "id": "loss-vs-bpb",
            "title": "Validation loss vs bits per byte",
            "description": "Separates pure token-level modeling loss from byte-normalized efficiency; color indicates vocabulary size.",
            "data": scatter_loss_vs_bpb(rows),
            "layout": {"xaxis": {"title": "Validation loss (nats/token)"}, "yaxis": {"title": "Bits per byte"}},
        },
        {
            "id": "val-bytes-token-vs-bpb",
            "title": "Validation bytes/token vs BPB",
            "description": "Shows whether BPB changes are driven by better byte coverage, better modeling, or both.",
            "data": metric_scatter(rows, "val_bytes_per_token", "bits_per_byte"),
            "layout": {"xaxis": {"title": "Validation bytes per token"}, "yaxis": {"title": "Bits per byte"}},
        },
        {
            "id": "byte-fallback-vs-bpb",
            "title": "Byte-fallback count vs BPB",
            "description": "Useful for spotting languages/vocab sizes where byte fallback may still dominate tokenization behavior.",
            "data": metric_scatter(rows, "unk_byte_fallback_tokens", "bits_per_byte"),
            "layout": {"xaxis": {"title": "Byte-fallback tokens"}, "yaxis": {"title": "Bits per byte"}},
        },
    ]


def render_html(rows: list[dict[str, str]], plots: list[dict[str, Any]]) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False)
    plots_json = json.dumps(plots, ensure_ascii=False)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>FLORES char-BPE validation and BPB report</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 2rem; color: #18212f; }}
    .plot-card {{ border: 1px solid #d6dbe3; border-radius: 12px; padding: 1rem; margin: 1.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .plot {{ width: 100%; min-height: 560px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #d6dbe3; padding: 0.35rem 0.5rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
    code {{ background: #f4f6f8; padding: 0.1rem 0.25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>FLORES char-BPE validation and bits-per-byte report</h1>
  <p>This report compares validation loss and BPB for char-BPE vocab sweeps. BPB is computed as <code>validation_loss_nats / (ln(2) * val_bytes_per_token)</code>.</p>
  <p>Total completed runs: <strong>{len(rows)}</strong></p>
  <div id=\"plots\"></div>
  <h2>Raw completed-run table</h2>
  <div id=\"table\"></div>
  <script>
    const rows = {rows_json};
    const plots = {plots_json};
    const container = document.getElementById('plots');
    for (const plot of plots) {{
      const card = document.createElement('section');
      card.className = 'plot-card';
      const title = document.createElement('h2');
      title.textContent = plot.title;
      const desc = document.createElement('p');
      desc.textContent = plot.description;
      const div = document.createElement('div');
      div.id = plot.id;
      div.className = 'plot';
      card.appendChild(title);
      card.appendChild(desc);
      card.appendChild(div);
      container.appendChild(card);
      Plotly.newPlot(div.id, plot.data, {{...plot.layout, margin: {{l: 90, r: 30, t: 30, b: 70}}, hovermode: 'closest'}}, {{responsive: true}});
    }}
    const tableColumns = ['language', 'vocab_size', 'validation_loss', 'bits_per_byte', 'val_bytes_per_token', 'unk_byte_fallback_tokens', 'out_dir'];
    const tableHtml = '<table><thead><tr>' + tableColumns.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>' +
      rows.map(row => '<tr>' + tableColumns.map(c => `<td>${{row[c] ?? ''}}</td>`).join('') + '</tr>').join('') +
      '</tbody></table>';
    document.getElementById('table').innerHTML = tableHtml;
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--tokenization-summary", type=Path, default=DEFAULT_TOKENIZATION_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    training_rows = read_csv_rows(args.training_summary)
    tokenization_rows = read_csv_rows(args.tokenization_summary) if args.tokenization_summary.exists() else []
    rows = enrich_training_rows(training_rows, tokenization_rows)
    plots = make_plots(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(rows, plots), encoding="utf-8")
    print(f"Wrote Plotly report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

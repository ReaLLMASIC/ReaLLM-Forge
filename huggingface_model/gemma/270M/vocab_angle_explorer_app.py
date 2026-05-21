#!/usr/bin/env python3
"""Dash webapp for Gemma vocab-angle neighborhood exploration."""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dash
from dash import Dash, Input, Output, State, dcc, html, no_update
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PRESETS = {
    "digits": [str(i) for i in range(10)],
    "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "months": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "alphabet": list("abcdefghijklmnopqrstuvwxyz"),
}


@dataclass
class ModelBundle:
    name: str
    tokenizer: AutoTokenizer
    emb: torch.Tensor
    emb_norm: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma vocab-angle explorer webapp")
    p.add_argument("--model-base", default="google/gemma-3-270m")
    p.add_argument("--model-it", default="google/gemma-3-270m-it")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--output-dir", default="./gemma_angle_explorer_exports")
    p.add_argument("--max-selected-tokens", type=int, default=2000)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _load_bundle(model_name: str, device: str, dtype: torch.dtype) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    emb = model.get_input_embeddings().weight.detach().to(device=device, dtype=dtype)
    emb_norm = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return ModelBundle(model_name, tokenizer, emb, emb_norm)


def _resolve_token_id(tokenizer: AutoTokenizer, text: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(text)
    if tid is not None and tid != tokenizer.unk_token_id:
        return int(tid)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return int(ids[0])


def _token_universe(bundle: ModelBundle) -> list[str]:
    return [bundle.tokenizer.convert_ids_to_tokens(i) for i in range(bundle.emb.size(0))]


def _apply_regex(tokens: Iterable[str], pattern: str) -> list[str]:
    if not pattern.strip():
        return list(tokens)
    rgx = re.compile(pattern)
    return [t for t in tokens if rgx.search(t)]


def _get_selection(base_tokens: list[str], preset: str, regex: str, max_selected: int) -> list[str]:
    if preset == "all":
        selected = list(base_tokens)
    else:
        selected = PRESETS.get(preset, PRESETS["digits"])[:]
    if regex.strip():
        selected = _apply_regex(selected if preset != "all" else base_tokens, regex)
    if len(selected) > max_selected:
        selected = selected[:max_selected]
    # preserve order + dedup
    out: list[str] = []
    seen = set()
    for t in selected:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _stack_counts_for_model(
    bundle: ModelBundle,
    token_texts: list[str],
    min_deg: float,
    max_deg: float,
    step_deg: float,
) -> tuple[list[str], np.ndarray, list[str], list[int]]:
    bins = np.arange(min_deg, max_deg + 1e-8, step_deg)
    if bins.size < 2:
        bins = np.array([min_deg, max_deg], dtype=np.float32)

    labels = [f"({bins[i]:.0f},{bins[i+1]:.0f}]" for i in range(len(bins) - 1)]
    token_ids: list[int] = []
    resolved_texts: list[str] = []
    for text in token_texts:
        tid = _resolve_token_id(bundle.tokenizer, text)
        if tid is None or tid >= bundle.emb.size(0):
            continue
        token_ids.append(tid)
        resolved_texts.append(text)

    if not token_ids:
        return [], np.zeros((0, len(labels)), dtype=np.int64), labels, []

    sel = bundle.emb_norm[token_ids]
    sims = torch.matmul(sel, bundle.emb_norm.T).cpu().numpy()
    sims = np.clip(sims, -1.0, 1.0)
    angles = np.degrees(np.arccos(sims))
    # exclude self match
    for i, tid in enumerate(token_ids):
        if 0 <= tid < angles.shape[1]:
            angles[i, tid] = 180.0

    counts = np.zeros((len(token_ids), len(labels)), dtype=np.int64)
    for b in range(len(labels)):
        lo, hi = bins[b], bins[b + 1]
        counts[:, b] = ((angles > lo) & (angles <= hi)).sum(axis=1)

    return resolved_texts, counts, labels, token_ids


def _sorted_order(token_texts: list[str], total_counts: np.ndarray, mode: str) -> np.ndarray:
    idx = np.arange(len(token_texts))
    if mode == "alphabetical":
        return np.array(sorted(idx, key=lambda i: token_texts[i].lower()))
    if mode == "highest_to_lowest":
        return np.argsort(-total_counts)
    return np.argsort(total_counts)


def _make_stack_figure(
    title: str,
    token_texts: list[str],
    counts: np.ndarray,
    labels: list[str],
    order: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    if len(token_texts) == 0:
        fig.update_layout(title=title)
        return fig

    ordered_tokens = [token_texts[i] for i in order]
    for b, lab in enumerate(labels):
        y = counts[order, b]
        fig.add_trace(go.Bar(x=ordered_tokens, y=y, name=lab, customdata=np.array(ordered_tokens)[:, None]))
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title="Token",
        yaxis_title="Count of vocab vectors in angle range",
        clickmode="event+select",
        margin=dict(l=40, r=20, t=50, b=120),
    )
    return fig


def _export_neighbors(
    bundle: ModelBundle,
    token_text: str,
    output_dir: Path,
) -> Path:
    tid = _resolve_token_id(bundle.tokenizer, token_text)
    if tid is None:
        raise ValueError(f"Could not resolve token: {token_text}")

    vec = bundle.emb[tid : tid + 1]
    vec_n = bundle.emb_norm[tid : tid + 1]
    dots = torch.matmul(vec, bundle.emb.T).squeeze(0).cpu().numpy()
    ndots = torch.matmul(vec_n, bundle.emb_norm.T).squeeze(0).cpu().numpy()
    ndots = np.clip(ndots, -1.0, 1.0)
    deg = np.degrees(np.arccos(ndots))

    out = output_dir / f"neighbors_{bundle.name.replace('/', '_')}_{tid}.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("token_id,token,degree_separation,dot_product,normalized_dot_product\n")
        for i in range(bundle.emb.size(0)):
            tok = bundle.tokenizer.convert_ids_to_tokens(i).replace("\n", "\\n")
            f.write(f"{i},{tok},{float(deg[i]):.8f},{float(dots[i]):.8f},{float(ndots[i]):.8f}\n")
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype(args.dtype)
    base = _load_bundle(args.model_base, args.device, dtype)
    it = _load_bundle(args.model_it, args.device, dtype)

    base_tokens = _token_universe(base)

    app: Dash = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Gemma Vocab Angle Explorer"),
            html.Div(
                [
                    html.Label("Preset"),
                    dcc.Dropdown(
                        id="preset",
                        options=[
                            {"label": "Digits", "value": "digits"},
                            {"label": "Weekdays", "value": "weekdays"},
                            {"label": "Months", "value": "months"},
                            {"label": "Alphabet", "value": "alphabet"},
                            {"label": "All", "value": "all"},
                        ],
                        value="digits",
                        clearable=False,
                    ),
                    html.Label("Regex filter"),
                    dcc.Input(id="regex", type="text", value="", style={"width": "100%"}),
                    html.Label("Min degrees"),
                    dcc.Input(id="min_deg", type="number", value=10, min=0, max=179, step=1),
                    html.Label("Max degrees"),
                    dcc.Input(id="max_deg", type="number", value=90, min=1, max=180, step=1),
                    html.Label("Stack step (degrees)"),
                    dcc.Input(id="step_deg", type="number", value=10, min=1, max=90, step=1),
                    html.Label("Sort"),
                    dcc.Dropdown(
                        id="sort_mode",
                        options=[
                            {"label": "Alphabetical", "value": "alphabetical"},
                            {"label": "Highest to lowest", "value": "highest_to_lowest"},
                            {"label": "Lowest to highest", "value": "lowest_to_highest"},
                        ],
                        value="alphabetical",
                        clearable=False,
                    ),
                ],
                style={"maxWidth": "420px"},
            ),
            html.Hr(),
            dcc.Graph(id="fig_base"),
            dcc.Graph(id="fig_it"),
            html.Div(id="clicked_token", style={"fontWeight": "bold"}),
            html.Button("Export neighbors CSV for clicked token", id="export_btn", n_clicks=0),
            html.Div(id="export_status"),
            dcc.Store(id="click_store"),
        ],
        style={"padding": "12px"},
    )

    @app.callback(
        Output("fig_base", "figure"),
        Output("fig_it", "figure"),
        Input("preset", "value"),
        Input("regex", "value"),
        Input("min_deg", "value"),
        Input("max_deg", "value"),
        Input("step_deg", "value"),
        Input("sort_mode", "value"),
    )
    def update_figs(preset: str, regex: str, min_deg: float, max_deg: float, step_deg: float, sort_mode: str):
        min_deg = float(min_deg or 10)
        max_deg = float(max_deg or 90)
        step_deg = max(1.0, float(step_deg or 10))
        if min_deg >= max_deg:
            max_deg = min_deg + 1

        selected = _get_selection(base_tokens, preset, regex or "", args.max_selected_tokens)
        t_base, c_base, labels_base, _ = _stack_counts_for_model(base, selected, min_deg, max_deg, step_deg)
        t_it, c_it, labels_it, _ = _stack_counts_for_model(it, selected, min_deg, max_deg, step_deg)

        totals = c_base.sum(axis=1) if c_base.size else np.array([])
        order = _sorted_order(t_base, totals, sort_mode) if len(t_base) else np.array([], dtype=int)

        fig_base = _make_stack_figure(f"{base.name} stacked histogram", t_base, c_base, labels_base, order)
        fig_it = _make_stack_figure(f"{it.name} stacked histogram", t_it, c_it, labels_it, order if len(t_it) == len(t_base) else np.arange(len(t_it)))
        return fig_base, fig_it

    @app.callback(
        Output("click_store", "data"),
        Output("clicked_token", "children"),
        Input("fig_base", "clickData"),
        Input("fig_it", "clickData"),
    )
    def capture_click(base_click, it_click):
        trigger = dash.callback_context.triggered_id
        click = base_click if trigger == "fig_base" else it_click
        if not click or not click.get("points"):
            return no_update, no_update
        token = click["points"][0]["x"]
        model_key = "base" if trigger == "fig_base" else "it"
        return {"token": token, "model": model_key}, f"Selected token: {token} (model={model_key})"

    @app.callback(
        Output("export_status", "children"),
        Input("export_btn", "n_clicks"),
        State("click_store", "data"),
        prevent_initial_call=True,
    )
    def export_clicked(_n, click_data):
        if not click_data:
            return "Click a token bar first."
        token = click_data["token"]
        model_key = click_data["model"]
        bundle = base if model_key == "base" else it
        out = _export_neighbors(bundle, token, output_dir)
        return f"Exported: {out}"

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

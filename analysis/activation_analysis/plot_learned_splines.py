#!/usr/bin/env python3
"""Plot learned spline activations from one or more checkpoints.

This script finds learned-spline knot parameters (`x_vals`, `y_vals`) in checkpoint
state dicts, reconstructs the spline curves, and writes a single interactive Plotly
HTML figure with one trace per layer/path.
"""

import argparse
import re
from pathlib import Path

import plotly.graph_objects as go
import torch


def _load_state_dict(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Common checkpoint layouts in this repo:
    # 1) {'model': state_dict, ...}
    # 2) raw state_dict
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt

    raise ValueError(f"Unsupported checkpoint format for: {checkpoint_path}")


def _natural_cubic_spline_eval(x_vals: torch.Tensor, y_vals: torch.Tensor, x_query: torch.Tensor):
    """Mirror LearnedSplineActivation's natural cubic spline evaluation."""
    x_vals = x_vals.to(dtype=torch.float64)
    y_vals = y_vals.to(dtype=torch.float64)
    x_query = x_query.to(dtype=torch.float64)

    n = x_vals.numel()
    if n < 2:
        raise ValueError("Need at least 2 knots to evaluate spline.")

    h = x_vals[1:] - x_vals[:-1]
    delta = (y_vals[1:] - y_vals[:-1]) / h

    a_mat = torch.zeros((n, n), dtype=torch.float64)
    rhs = torch.zeros((n,), dtype=torch.float64)

    a_mat[0, 0] = 1.0
    a_mat[-1, -1] = 1.0

    for i in range(1, n - 1):
        a_mat[i, i - 1] = h[i - 1]
        a_mat[i, i] = 2 * (h[i - 1] + h[i])
        a_mat[i, i + 1] = h[i]
        rhs[i] = 3 * (delta[i] - delta[i - 1])

    m_vals = torch.linalg.solve(a_mat, rhs)

    # Interval coefficients
    a_coef = y_vals[:-1]
    b_coef = delta - h * (2 * m_vals[:-1] + m_vals[1:]) / 3
    c_coef = m_vals[:-1] / 2
    d_coef = (m_vals[1:] - m_vals[:-1]) / (6 * h)

    indices = torch.searchsorted(x_vals, x_query, right=False) - 1
    indices = torch.clamp(indices, 0, n - 2)

    x_k = x_vals[indices]
    dx = x_query - x_k

    return (
        a_coef[indices]
        + b_coef[indices] * dx
        + c_coef[indices] * dx**2
        + d_coef[indices] * dx**3
    )


def _extract_layer_index(param_name: str):
    match = re.search(r"\.h\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))
    return None


def _collect_learned_splines(state_dict: dict, checkpoint_path: Path):
    traces = []

    # LearnedSplineActivation in this codebase stores parameters as:
    # ...activation_variant.x_vals and ...activation_variant.y_vals
    x_keys = [k for k in state_dict if k.endswith(".x_vals") and "activation_variant" in k]

    for x_key in x_keys:
        y_key = x_key[:-len(".x_vals")] + ".y_vals"
        if y_key not in state_dict:
            continue

        x_vals = state_dict[x_key]
        y_vals = state_dict[y_key]
        if not isinstance(x_vals, torch.Tensor) or not isinstance(y_vals, torch.Tensor):
            continue
        if x_vals.ndim != 1 or y_vals.ndim != 1:
            continue
        if x_vals.numel() != y_vals.numel() or x_vals.numel() < 4:
            continue

        layer_idx = _extract_layer_index(x_key)
        traces.append(
            {
                "checkpoint": str(checkpoint_path),
                "layer_idx": layer_idx,
                "x_key": x_key,
                "x_vals": x_vals.detach().cpu(),
                "y_vals": y_vals.detach().cpu(),
            }
        )

    traces.sort(key=lambda t: (t["checkpoint"], -1 if t["layer_idx"] is None else t["layer_idx"], t["x_key"]))
    return traces


def _build_html_with_range_controls(fig, html_path: Path, default_xmin: float, default_xmax: float):
    div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="spline_plot")

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Learned spline curves</title>
  <style>
    body {{ font-family: sans-serif; margin: 18px; }}
    .controls {{ margin-bottom: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    input {{ width: 110px; }}
    .hint {{ color: #555; }}
  </style>
</head>
<body>
  <h2>Learned spline curves</h2>
  <div class=\"controls\">
    <label for=\"xmin\">x-min</label>
    <input id=\"xmin\" type=\"number\" step=\"0.1\" value=\"{default_xmin}\" />
    <label for=\"xmax\">x-max</label>
    <input id=\"xmax\" type=\"number\" step=\"0.1\" value=\"{default_xmax}\" />
    <button id=\"apply_range\">Apply x-range</button>
    <button id=\"reset_range\">Reset default</button>
    <span class=\"hint\">Legend labels include checkpoint path + layer + parameter path.</span>
  </div>
  {div}

  <script>
    const gd = document.getElementById('spline_plot');
    const xminInput = document.getElementById('xmin');
    const xmaxInput = document.getElementById('xmax');
    const applyBtn = document.getElementById('apply_range');
    const resetBtn = document.getElementById('reset_range');

    function applyRange() {{
      const xmin = Number(xminInput.value);
      const xmax = Number(xmaxInput.value);
      if (!Number.isFinite(xmin) || !Number.isFinite(xmax) || xmin >= xmax) {{
        alert('Please provide valid x-min and x-max values where x-min < x-max.');
        return;
      }}
      Plotly.relayout(gd, {{'xaxis.range': [xmin, xmax]}});
    }}

    applyBtn.addEventListener('click', applyRange);
    resetBtn.addEventListener('click', () => {{
      xminInput.value = {default_xmin};
      xmaxInput.value = {default_xmax};
      applyRange();
    }});
  </script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot learned spline activations from checkpoints.")
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Checkpoint path(s), e.g. out/run1/ckpt.pt out/run2/ckpt.pt",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=Path("analysis/activation_analysis/learned_splines.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--x_min", type=float, default=-10.0, help="Default plot x-axis minimum.")
    parser.add_argument("--x_max", type=float, default=10.0, help="Default plot x-axis maximum.")
    parser.add_argument("--num_points", type=int, default=1200, help="Points per spline trace.")
    args = parser.parse_args()

    if args.x_min >= args.x_max:
        raise ValueError("x_min must be smaller than x_max")
    if args.num_points < 8:
        raise ValueError("num_points should be >= 8")

    x_query = torch.linspace(args.x_min, args.x_max, args.num_points, dtype=torch.float64)

    figure = go.Figure()
    total_splines = 0

    for checkpoint_path in args.checkpoints:
        state_dict = _load_state_dict(checkpoint_path)
        splines = _collect_learned_splines(state_dict, checkpoint_path)

        for spline in splines:
            y_query = _natural_cubic_spline_eval(spline["x_vals"], spline["y_vals"], x_query)
            layer_label = f"layer {spline['layer_idx']}" if spline["layer_idx"] is not None else "layer ?"
            label = f"{spline['checkpoint']} | {layer_label} | {spline['x_key']}"

            figure.add_trace(
                go.Scatter(
                    x=x_query.tolist(),
                    y=y_query.tolist(),
                    mode="lines",
                    name=label,
                    hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra>%{fullData.name}</extra>",
                )
            )
            total_splines += 1

    if total_splines == 0:
        raise RuntimeError(
            "No learned spline parameters were found. "
            "Expected keys like '...activation_variant.x_vals' + '...activation_variant.y_vals'."
        )

    figure.update_layout(
        title=f"Learned spline curves ({total_splines} traces)",
        xaxis_title="x",
        yaxis_title="activation(x)",
        template="plotly_white",
        legend=dict(orientation="v"),
    )
    figure.update_xaxes(range=[args.x_min, args.x_max])

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    _build_html_with_range_controls(figure, args.output_html, args.x_min, args.x_max)
    print(f"Saved: {args.output_html} ({total_splines} traces)")


if __name__ == "__main__":
    main()

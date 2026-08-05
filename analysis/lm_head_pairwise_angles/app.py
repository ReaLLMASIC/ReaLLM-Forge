#!/usr/bin/env python3
"""Small Flask webapp for LM-head pairwise angle checkpoint comparisons."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_lm_head_pairwise_angles import compare_lm_head_pairwise_angles, plot_html


HERO_METRICS = [
    ("angle_alignment_score_pct", "Alignment", "%"),
    ("pearson_r", "Pearson r", ""),
    ("mae_deg", "MAE", "°"),
    ("rmse_deg", "RMSE", "°"),
    ("selected_pairs", "Selected pairs", ""),
]


def find_checkpoints(root: str, limit: int = 500) -> list[str]:
    root_path = Path(root).resolve()
    paths = []
    for path in root_path.rglob("*.pt"):
        if path.is_file():
            paths.append(str(path))
            if len(paths) >= limit:
                break
    return sorted(paths)


def fmt_metric(value: float, suffix: str = "") -> str:
    if value != value:  # NaN check without importing math
        return "n/a"
    if abs(value) >= 1000 and suffix != "%":
        text = f"{value:,.0f}"
    elif suffix == "%":
        text = f"{value:.2f}"
    else:
        text = f"{value:.4g}"
    return f"{text}{suffix}"


def create_app(ckpt_root: str = "."):
    try:
        from flask import Flask, request
    except ImportError as exc:
        raise SystemExit("Flask is required for the webapp. Install it with `pip install flask`.") from exc

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        ckpts = find_checkpoints(ckpt_root)
        default_a = ckpts[0] if ckpts else ""
        default_b = ckpts[1] if len(ckpts) > 1 else default_a
        result_html = ""
        error = ""
        form = request.form if request.method == "POST" else {}
        ckpt_a = form.get("ckpt_a", default_a)
        ckpt_b = form.get("ckpt_b", default_b)
        meta = form.get("meta", "")
        device = form.get("device", "cpu")
        min_angle = float(form.get("min_angle", 0.0) or 0.0)
        max_angle = float(form.get("max_angle", 180.0) or 180.0)
        if request.method == "POST":
            try:
                result = compare_lm_head_pairwise_angles(
                    ckpt_a,
                    ckpt_b,
                    meta=meta or None,
                    device=device,
                    min_angle=min_angle,
                    max_angle=max_angle,
                )
                metric_cards = "".join(
                    f"<div class='metric-card'><span>{label}</span><strong>{fmt_metric(result.metrics.get(key, float('nan')), suffix)}</strong></div>"
                    for key, label, suffix in HERO_METRICS
                )
                metrics = "".join(
                    f"<tr><th>{html.escape(k)}</th><td>{fmt_metric(v)}</td></tr>"
                    for k, v in result.metrics.items()
                )
                result_html = f"""
                <section class='results'>
                  <div class='section-heading'>
                    <div>
                      <p class='eyebrow'>Baseline alignment</p>
                      <h2>Pairwise-angle preservation vs checkpoint A</h2>
                    </div>
                    <p class='hint'>The alignment scatter compares every selected pair's checkpoint-A angle to checkpoint-B angle. The dashed diagonal is perfect preservation.</p>
                  </div>
                  <div class='metrics'>{metric_cards}</div>
                  <details class='metrics-table'>
                    <summary>Show all metrics</summary>
                    <table>{metrics}</table>
                  </details>
                  <div class='plot-shell'>{plot_html(result)}</div>
                </section>
                """
            except Exception as exc:  # render user-facing analysis errors in the page
                error = f"<pre class='error'>{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</pre>"
        options = "".join(
            f"<option value='{html.escape(p, quote=True)}' {'selected' if p == ckpt_a else ''}>{html.escape(p)}</option>"
            for p in ckpts
        )
        options_b = "".join(
            f"<option value='{html.escape(p, quote=True)}' {'selected' if p == ckpt_b else ''}>{html.escape(p)}</option>"
            for p in ckpts
        )
        meta_value = html.escape(meta or "data/shakespeare_char/meta.pkl", quote=True)
        device_options = "".join(
            f"<option {'selected' if device == item else ''}>{item}</option>" for item in ["cpu", "auto", "cuda"]
        )
        return f"""<!doctype html><html><head><title>LM head pairwise angles</title>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
:root{{color-scheme:dark;--bg:#070b1a;--panel:rgba(15,23,42,.82);--panel2:rgba(30,41,59,.72);--text:#e5eefc;--muted:#93a4bd;--accent:#7c3aed;--accent2:#06b6d4;--line:rgba(148,163,184,.25)}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:radial-gradient(circle at top left,#1e1b4b 0,#0f172a 34%,#030712 100%);color:var(--text)}}
.hero{{padding:48px 56px 28px;background:linear-gradient(135deg,rgba(124,58,237,.24),rgba(6,182,212,.12));border-bottom:1px solid var(--line)}}
.eyebrow{{margin:0 0 8px;color:#67e8f9;text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:800}} h1{{font-size:clamp(34px,5vw,64px);line-height:1;margin:0 0 14px}} h2{{margin:.1rem 0 .4rem;font-size:26px}} .hero p{{max-width:920px;color:var(--muted);font-size:18px}}
.container{{padding:28px 56px 56px}} .control-card,.results{{background:var(--panel);border:1px solid var(--line);border-radius:24px;box-shadow:0 24px 80px rgba(0,0,0,.35);backdrop-filter:blur(16px);padding:24px;margin-bottom:28px}}
.form-grid{{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:18px}} label{{display:flex;flex-direction:column;gap:8px;color:#cbd5e1;font-weight:700}} input,select{{width:100%;border:1px solid var(--line);background:rgba(2,6,23,.78);color:var(--text);border-radius:14px;padding:12px 14px;outline:none}} input:focus,select:focus{{border-color:#22d3ee;box-shadow:0 0 0 3px rgba(34,211,238,.16)}}
.actions{{display:flex;align-items:center;gap:14px;margin-top:20px}} button{{border:0;border-radius:999px;padding:13px 24px;font-weight:900;color:white;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 12px 36px rgba(6,182,212,.25);cursor:pointer}} .hint{{color:var(--muted);font-size:14px}}
.section-heading{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:18px}} .metrics{{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:14px;margin:18px 0}} .metric-card{{background:linear-gradient(180deg,rgba(30,41,59,.92),rgba(15,23,42,.92));border:1px solid var(--line);border-radius:18px;padding:18px}} .metric-card span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .metric-card strong{{font-size:28px;line-height:1.3}}
.metrics-table{{margin:16px 0;color:var(--muted)}} summary{{cursor:pointer;font-weight:800;color:#bfdbfe}} table{{border-collapse:collapse;margin-top:12px;width:100%;overflow:hidden;border-radius:12px}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}} th{{color:#cbd5e1}} .plot-shell{{background:#fff;border-radius:18px;padding:10px;overflow:auto}} .error{{white-space:pre-wrap;background:rgba(127,29,29,.28);border:1px solid rgba(248,113,113,.45);border-radius:16px;color:#fecaca;padding:16px}}
@media (max-width:900px){{.hero,.container{{padding-left:22px;padding-right:22px}}.form-grid,.metrics{{grid-template-columns:1fr}}.section-heading{{display:block}}}}
</style></head><body>
<header class='hero'><p class='eyebrow'>nanoGPT geometry lab</p><h1>LM-head pairwise angle comparison</h1><p>Compare the vocabulary-vector geometry of two checkpoints, inspect preservation against checkpoint A as the baseline, and explore pairwise angles with CUDA acceleration when available.</p></header>
<main class='container'>
<section class='control-card'>
<form method='post'>
<div class='form-grid'>
<label>Baseline checkpoint A <select name='ckpt_a'>{options}</select></label>
<label>Comparison checkpoint B <select name='ckpt_b'>{options_b}</select></label>
<label>Meta pickle for token labels <input name='meta' value='{meta_value}'></label>
<label>Device <select name='device'>{device_options}</select></label>
<label>Minimum baseline-A pair angle in degrees <input type='number' step='0.1' name='min_angle' value='{min_angle}'></label>
<label>Maximum baseline-A pair angle in degrees <input type='number' step='0.1' name='max_angle' value='{max_angle}'></label>
</div>
<div class='actions'><button type='submit'>Compare geometry</button><span class='hint'>Found {len(ckpts)} checkpoint file(s) under {html.escape(str(Path(ckpt_root).resolve()))}</span></div>
</form>
</section>
{error}{result_html}
</main></body></html>"""

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_app(args.ckpt_root).run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()

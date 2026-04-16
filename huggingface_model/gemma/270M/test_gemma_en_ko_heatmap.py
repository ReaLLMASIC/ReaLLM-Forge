# test_gemma_en_ko_heatmap.py
"""Test Gemma 270M tokenization on English-Korean translation pairs with heatmaps.

Fetches EN-KO translation pairs from the OPUS/Helsinki-NLP dataset via
HuggingFace, tokenizes both sides with the Gemma tokenizer, and produces:

  1. **Terminal heatmap** – ANSI-coloured token grids showing token density,
     byte-per-token ratio, and character-per-token ratio.
  2. **Interactive HTML heatmap** – a self-contained page with hover tooltips,
     side-by-side EN/KO comparison, and switchable colour modes.

With ``--inference``, loads the full Gemma 270M model and adds three
colour modes based on next-token prediction:

  * **rank**        – green = rank 1 (model's top pick), red = rank ≥ ``--rank_red``
  * **probability** – green = 1.0 softmax confidence, red = 0.0
  * **minmax**      – red = lowest prob in the sentence, green = highest

Requirements: transformers, datasets, rich (optional for terminal)

Usage
-----
```bash
# quick test – 5 sentence pairs, terminal only
python test_gemma_en_ko_heatmap.py

# with inference heatmap – 10 pairs
python test_gemma_en_ko_heatmap.py \
    --num_pairs 10 --inference \
    --html en_ko_heatmap.html \
    --output_file en_ko_terminal.txt
```
"""
from __future__ import annotations

import argparse
import colorsys
import html as html_lib
import io
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------

def _hue_hex(hue: float, s: float = 0.7, l: float = 0.5) -> str:
    r, g, b = colorsys.hls_to_rgb(hue, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _rg_gradient(norm: float) -> str:
    """Red (0) -> Green (1)."""
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


def _scalar_colour(value: float, lo: float, hi: float) -> str:
    if hi - lo < 1e-9:
        norm = 0.5
    else:
        norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def tokenize_text(text: str, tokenizer) -> List[Tuple[int, str, str]]:
    """Return [(token_id, token_string, original_segment), ...]."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        spans.append((tid, tok_str, text[s:e]))
    return spans


def compute_metrics(spans: List[Tuple[int, str, str]]) -> Dict[str, float]:
    """Aggregate tokenization metrics."""
    if not spans:
        return {"num_tokens": 0, "total_chars": 0, "total_bytes": 0,
                "chars_per_token": 0, "bytes_per_token": 0}
    total_chars = sum(len(o) for _, _, o in spans)
    total_bytes = sum(len(o.encode("utf-8")) for _, _, o in spans)
    n = len(spans)
    return {
        "num_tokens": n,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "chars_per_token": total_chars / n,
        "bytes_per_token": total_bytes / n,
    }


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def run_inference(
    token_ids: List[int],
    model,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """For each token at position i (i>=1), compute rank and probability.

    Position 0 gets rank=0, probability=1.0 (no prior context).
    """
    import torch
    import torch.nn.functional as F

    results: List[Dict[str, float]] = [{"rank": 0, "probability": 1.0}]

    if len(token_ids) <= 1:
        return results

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits.squeeze(0)  # (seq_len, vocab)

    for i in range(1, len(token_ids)):
        step_logits = logits[i - 1]
        probs = F.softmax(step_logits, dim=-1)
        target_id = token_ids[i]
        prob = probs[target_id].item()
        rank = int((step_logits > step_logits[target_id]).sum().item()) + 1
        results.append({"rank": rank, "probability": prob})

    return results


def run_inference_batch(
    all_spans: List[List[Tuple[int, str, str]]],
    model,
    device: str = "cpu",
) -> List[List[Dict[str, float]]]:
    """Run inference on multiple texts, returning results per text."""
    results = []
    for spans in all_spans:
        token_ids = [tid for tid, _, _ in spans]
        results.append(run_inference(token_ids, model, device))
    return results


# ---------------------------------------------------------------------------
# fetch EN-KO pairs
# ---------------------------------------------------------------------------

def fetch_en_ko_pairs(num_pairs: int, dataset_name: str, dataset_config: str) -> List[Tuple[str, str]]:
    """Load EN-KO translation pairs from HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} ({dataset_config})...")
    ds = load_dataset(dataset_name, dataset_config, split=f"train[:{num_pairs}]")
    pairs = []
    for row in ds:
        tr = row["translation"]
        pairs.append((tr["en"], tr["ko"]))
    return pairs


# ---------------------------------------------------------------------------
# terminal rendering
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _colour_spans_terminal(
    spans: List[Tuple[int, str, str]],
    mode: str,
    inference_results: Optional[List[Dict[str, float]]] = None,
    rank_red: int = 100,
) -> List[str]:
    """Return hex colours for terminal based on mode."""
    if mode == "byte_length":
        bls = [len(o.encode("utf-8")) for _, _, o in spans]
        lo, hi = (min(bls), max(bls)) if bls else (0, 1)
        return [_rg_gradient(1 - (bl - lo) / (hi - lo + 1e-9)) for bl in bls]

    if inference_results is None:
        # fallback to byte_length
        bls = [len(o.encode("utf-8")) for _, _, o in spans]
        lo, hi = (min(bls), max(bls)) if bls else (0, 1)
        return [_rg_gradient(1 - (bl - lo) / (hi - lo + 1e-9)) for bl in bls]

    if mode == "rank":
        colours = []
        for res in inference_results:
            if res["rank"] == 0:
                colours.append("#888888")
            else:
                norm = 1.0 - (min(res["rank"], rank_red) - 1) / max(rank_red - 1, 1)
                colours.append(_rg_gradient(norm))
        return colours

    if mode == "probability":
        return [
            "#888888" if r["rank"] == 0 else _rg_gradient(r["probability"])
            for r in inference_results
        ]

    if mode == "minmax":
        probs = [r["probability"] for r in inference_results if r["rank"] > 0]
        if not probs:
            return ["#888888"] * len(spans)
        lo, hi = min(probs), max(probs)
        return [
            "#888888" if r["rank"] == 0 else _scalar_colour(r["probability"], lo, hi)
            for r in inference_results
        ]

    return ["#888888"] * len(spans)


def render_terminal_heatmap(
    pairs: List[Tuple[str, str]],
    en_all: List[List[Tuple[int, str, str]]],
    ko_all: List[List[Tuple[int, str, str]]],
    colour_mode: str = "byte_length",
    en_inf: Optional[List[List[Dict[str, float]]]] = None,
    ko_inf: Optional[List[List[Dict[str, float]]]] = None,
    rank_red: int = 100,
) -> str:
    """Render side-by-side EN/KO heatmaps to terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        lines = []
        for i, (en, ko) in enumerate(pairs):
            lines.append(f"--- Pair {i+1} ---")
            lines.append(f"EN: {en}")
            lines.append(f"KO: {ko}")
            lines.append(f"EN tokens: {len(en_all[i])}  KO tokens: {len(ko_all[i])}")
            lines.append("")
        return "\n".join(lines)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)

    # Summary table
    title = "EN-KO Tokenization Comparison (Gemma 270M)"
    if en_inf is not None:
        title += f" – colour: {colour_mode}"
    summary = Table(title=title, box=None, pad_edge=False)
    summary.add_column("#", justify="right", style="dim")
    summary.add_column("EN tokens", justify="right")
    summary.add_column("KO tokens", justify="right")
    summary.add_column("Ratio KO/EN", justify="right")
    summary.add_column("EN bytes/tok", justify="right")
    summary.add_column("KO bytes/tok", justify="right")
    if en_inf is not None:
        summary.add_column("EN avg rank", justify="right")
        summary.add_column("KO avg rank", justify="right")

    for i, (en_spans, ko_spans) in enumerate(zip(en_all, ko_all)):
        en_m = compute_metrics(en_spans)
        ko_m = compute_metrics(ko_spans)
        ratio = ko_m["num_tokens"] / max(en_m["num_tokens"], 1)
        ratio_col = _rg_gradient(1 - min(ratio / 3, 1))

        row = [
            str(i + 1),
            str(en_m["num_tokens"]),
            str(ko_m["num_tokens"]),
            Text(f"{ratio:.2f}", style=f"bold {ratio_col}"),
            f"{en_m['bytes_per_token']:.1f}",
            f"{ko_m['bytes_per_token']:.1f}",
        ]
        if en_inf is not None:
            en_ranks = [r["rank"] for r in en_inf[i] if r["rank"] > 0]
            ko_ranks = [r["rank"] for r in ko_inf[i] if r["rank"] > 0]
            en_avg = sum(en_ranks) / len(en_ranks) if en_ranks else 0
            ko_avg = sum(ko_ranks) / len(ko_ranks) if ko_ranks else 0
            en_col = _rg_gradient(1 - min(en_avg / rank_red, 1))
            ko_col = _rg_gradient(1 - min(ko_avg / rank_red, 1))
            row.append(Text(f"{en_avg:.1f}", style=f"bold {en_col}"))
            row.append(Text(f"{ko_avg:.1f}", style=f"bold {ko_col}"))

        summary.add_row(*row)

    console.print(summary)
    console.print()

    # Per-pair token heatmaps
    for i, ((en_text, ko_text), en_spans, ko_spans) in enumerate(
        zip(pairs, en_all, ko_all)
    ):
        console.print(f"[bold cyan]--- Pair {i+1} ---[/bold cyan]")

        en_colours = _colour_spans_terminal(
            en_spans, colour_mode,
            inference_results=en_inf[i] if en_inf else None,
            rank_red=rank_red,
        )
        ko_colours = _colour_spans_terminal(
            ko_spans, colour_mode,
            inference_results=ko_inf[i] if ko_inf else None,
            rank_red=rank_red,
        )

        en_line = Text()
        for (tid, tok, orig), col in zip(en_spans, en_colours):
            en_line.append(f"|{_escape(orig)}", style=f"bold {col}")
        en_line.append("|", style="dim")

        ko_line = Text()
        for (tid, tok, orig), col in zip(ko_spans, ko_colours):
            ko_line.append(f"|{_escape(orig)}", style=f"bold {col}")
        ko_line.append("|", style="dim")

        console.print(Text("EN: ", style="bold"), end="")
        console.print(en_line)
        console.print(Text("KO: ", style="bold"), end="")
        console.print(ko_line)
        console.print()

    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _js_str(s: str) -> str:
    return (
        '"'
        + s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("'", "\\'")
        + '"'
    )


def render_html_heatmap(
    pairs: List[Tuple[str, str]],
    en_all: List[List[Tuple[int, str, str]]],
    ko_all: List[List[Tuple[int, str, str]]],
    title: str = "Gemma 270M – EN↔KO Tokenization Heatmap",
    en_inf: Optional[List[List[Dict[str, float]]]] = None,
    ko_inf: Optional[List[List[Dict[str, float]]]] = None,
    rank_red: int = 100,
) -> str:
    """Build interactive HTML with side-by-side heatmaps."""
    has_inference = en_inf is not None

    # Build JS data
    pair_data_js = []
    for i, ((en_text, ko_text), en_spans, ko_spans) in enumerate(
        zip(pairs, en_all, ko_all)
    ):
        def _tok_js(spans, inf_list):
            parts = []
            for j, (tid, tok, orig) in enumerate(spans):
                inf = inf_list[j] if inf_list else {"rank": -1, "probability": -1}
                parts.append(
                    f'{{id:{tid},tok:{_js_str(tok)},orig:{_js_str(orig)},'
                    f'chars:{len(orig)},bytes:{len(orig.encode("utf-8"))},'
                    f'rank:{inf["rank"]},prob:{inf["probability"]:.6f}}}'
                )
            return ",".join(parts)

        en_inf_i = en_inf[i] if en_inf else None
        ko_inf_i = ko_inf[i] if ko_inf else None
        pair_data_js.append(
            f'{{en_text:{_js_str(en_text)},ko_text:{_js_str(ko_text)},'
            f'en_tokens:[{_tok_js(en_spans, en_inf_i)}],'
            f'ko_tokens:[{_tok_js(ko_spans, ko_inf_i)}]}}'
        )

    inference_options = ""
    if has_inference:
        inference_options = """
      <option value="rank">Rank (green=top-1, red=low)</option>
      <option value="probability">Probability (green=1.0, red=0.0)</option>
      <option value="minmax">MinMax probability (relative)</option>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#e6edf3; padding:20px; }}
  h1 {{ font-size:1.5em; margin-bottom:8px; color:#58a6ff; }}
  .subtitle {{ color:#8b949e; margin-bottom:20px; font-size:0.9em; }}
  .controls {{ margin-bottom:16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
  .controls label {{ font-size:0.85em; color:#8b949e; }}
  .controls select {{ background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 10px; }}
  .pair {{ background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  .pair-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
  .pair-header h3 {{ font-size:1em; color:#58a6ff; }}
  .pair-stats {{ font-size:0.8em; color:#8b949e; }}
  .lang-row {{ margin-bottom:8px; }}
  .lang-label {{ display:inline-block; width:28px; font-weight:bold; color:#f0883e; font-size:0.85em; vertical-align:top; padding-top:4px; }}
  .token-row {{ display:inline; line-height:2.2; }}
  .tok {{
    display:inline-block; padding:2px 5px; margin:1px; border-radius:3px;
    cursor:pointer; font-family:'Fira Code','Consolas',monospace; font-size:0.9em;
    border:1px solid rgba(255,255,255,0.08); transition:transform 0.1s;
  }}
  .tok:hover {{ transform:scale(1.15); box-shadow:0 0 10px rgba(88,166,255,0.4); z-index:10; position:relative; }}
  #tooltip {{
    display:none; position:fixed; background:#1f2937; color:#fff;
    padding:10px 14px; border-radius:6px; font-size:0.82em; z-index:100;
    box-shadow:0 4px 16px rgba(0,0,0,0.6); max-width:400px;
    font-family:'Fira Code',monospace; line-height:1.7; border:1px solid #30363d;
  }}
  #tooltip .label {{ color:#58a6ff; }}
  .summary-table {{ width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.85em; }}
  .summary-table th {{ text-align:left; padding:8px; border-bottom:2px solid #30363d; color:#58a6ff; }}
  .summary-table td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
  .summary-table tr:hover {{ background:#21262d; }}
  .ratio-badge {{
    display:inline-block; padding:2px 8px; border-radius:10px;
    font-weight:bold; font-size:0.85em;
  }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="subtitle">Hover over tokens for details. Korean text typically requires more tokens than English for the same semantic content.</p>

<div class="controls">
  <label>Colour by:
    <select id="colourMode">
      <option value="byte_length">Byte length (heatmap)</option>
      <option value="char_length">Char length</option>
      <option value="token_id">Token ID (hashed)</option>{inference_options}
    </select>
  </label>
</div>

<table class="summary-table" id="summary"></table>

<div id="pairs-container"></div>
<div id="tooltip"></div>

<script>
const data = [{",".join(pair_data_js)}];
const RANK_RED = {rank_red};
const HAS_INFERENCE = {'true' if has_inference else 'false'};
const container = document.getElementById('pairs-container');
const tooltip = document.getElementById('tooltip');
const golden = 0.618033988749895;

function hueCol(id) {{
  const h = (id * golden) % 1.0;
  const s = 0.65, l = 0.5;
  const f = (n) => {{
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round((l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))) * 255);
  }};
  return '#' + [f(0),f(8),f(4)].map(x => x.toString(16).padStart(2,'0')).join('');
}}

function rgGrad(norm) {{
  const n = Math.max(0, Math.min(1, norm));
  const r = Math.round((1-n)*255), g = Math.round(n*255);
  return '#' + r.toString(16).padStart(2,'0') + g.toString(16).padStart(2,'0') + '00';
}}

function scalarCol(val, lo, hi) {{
  const n = hi-lo < 1e-9 ? 0.5 : Math.max(0, Math.min(1, (val-lo)/(hi-lo)));
  return rgGrad(n);
}}

function fgFor(bg) {{
  const r=parseInt(bg.slice(1,3),16), g=parseInt(bg.slice(3,5),16), b=parseInt(bg.slice(5,7),16);
  return (0.299*r+0.587*g+0.114*b) > 140 ? '#000' : '#fff';
}}

function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function colourTokens(tokens, mode) {{
  if (mode === 'token_id') return tokens.map(t => hueCol(t.id));
  if (mode === 'char_length' || mode === 'byte_length') {{
    const key = mode === 'char_length' ? 'chars' : 'bytes';
    const vals = tokens.map(t => t[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return vals.map(v => rgGrad(1 - (hi-lo < 1e-9 ? 0.5 : (v-lo)/(hi-lo))));
  }}
  if (mode === 'rank') {{
    return tokens.map(t => {{
      if (t.rank === 0) return '#888888';
      return rgGrad(1.0 - (Math.min(t.rank, RANK_RED) - 1) / Math.max(RANK_RED - 1, 1));
    }});
  }}
  if (mode === 'probability') {{
    return tokens.map(t => t.rank === 0 ? '#888888' : rgGrad(t.prob));
  }}
  if (mode === 'minmax') {{
    const scored = tokens.filter(t => t.rank > 0);
    if (scored.length === 0) return tokens.map(() => '#888888');
    const probs = scored.map(t => t.prob);
    const lo = Math.min(...probs), hi = Math.max(...probs);
    return tokens.map(t => t.rank === 0 ? '#888888' : scalarCol(t.prob, lo, hi));
  }}
  return tokens.map(() => '#888888');
}}

function renderSummary() {{
  const tbl = document.getElementById('summary');
  let cols = '<tr><th>#</th><th>EN tokens</th><th>KO tokens</th><th>Ratio KO/EN</th><th>EN bytes/tok</th><th>KO bytes/tok</th>';
  if (HAS_INFERENCE) cols += '<th>EN avg rank</th><th>KO avg rank</th><th>EN avg prob</th><th>KO avg prob</th>';
  cols += '</tr>';
  let html = cols;
  data.forEach((p, i) => {{
    const enN = p.en_tokens.length, koN = p.ko_tokens.length;
    const ratio = koN / Math.max(enN, 1);
    const enB = p.en_tokens.reduce((s,t)=>s+t.bytes,0) / Math.max(enN,1);
    const koB = p.ko_tokens.reduce((s,t)=>s+t.bytes,0) / Math.max(koN,1);
    const col = rgGrad(1 - Math.min(ratio/3, 1));
    html += `<tr><td>${{i+1}}</td><td>${{enN}}</td><td>${{koN}}</td>` +
      `<td><span class="ratio-badge" style="background:${{col}};color:${{fgFor(col)}}">${{ratio.toFixed(2)}}</span></td>` +
      `<td>${{enB.toFixed(1)}}</td><td>${{koB.toFixed(1)}}</td>`;
    if (HAS_INFERENCE) {{
      const enS = p.en_tokens.filter(t=>t.rank>0), koS = p.ko_tokens.filter(t=>t.rank>0);
      const enAvgR = enS.length ? (enS.reduce((s,t)=>s+t.rank,0)/enS.length) : 0;
      const koAvgR = koS.length ? (koS.reduce((s,t)=>s+t.rank,0)/koS.length) : 0;
      const enAvgP = enS.length ? (enS.reduce((s,t)=>s+t.prob,0)/enS.length) : 0;
      const koAvgP = koS.length ? (koS.reduce((s,t)=>s+t.prob,0)/koS.length) : 0;
      const erc = rgGrad(1-Math.min(enAvgR/RANK_RED,1)), krc = rgGrad(1-Math.min(koAvgR/RANK_RED,1));
      html += `<td><span class="ratio-badge" style="background:${{erc}};color:${{fgFor(erc)}}">${{enAvgR.toFixed(1)}}</span></td>`;
      html += `<td><span class="ratio-badge" style="background:${{krc}};color:${{fgFor(krc)}}">${{koAvgR.toFixed(1)}}</span></td>`;
      html += `<td>${{enAvgP.toFixed(4)}}</td><td>${{koAvgP.toFixed(4)}}</td>`;
    }}
    html += '</tr>';
  }});
  tbl.innerHTML = html;
}}

function renderPairs() {{
  const mode = document.getElementById('colourMode').value;
  container.innerHTML = '';
  data.forEach((p, i) => {{
    const div = document.createElement('div');
    div.className = 'pair';

    const enCols = colourTokens(p.en_tokens, mode);
    const koCols = colourTokens(p.ko_tokens, mode);
    const enN = p.en_tokens.length, koN = p.ko_tokens.length;
    const ratio = (koN / Math.max(enN, 1)).toFixed(2);

    let html = `<div class="pair-header"><h3>Pair ${{i+1}}</h3>` +
      `<span class="pair-stats">EN: ${{enN}} tokens | KO: ${{koN}} tokens | ratio: ${{ratio}}</span></div>`;

    html += renderLangRow('EN', p.en_tokens, enCols);
    html += renderLangRow('KO', p.ko_tokens, koCols);
    div.innerHTML = html;
    container.appendChild(div);
  }});

  // Attach hover events
  container.querySelectorAll('.tok').forEach(el => {{
    el.addEventListener('mouseenter', (ev) => {{
      const d = el.dataset;
      let html =
        `<span class="label">Token ID:</span> ${{d.tid}}<br>` +
        `<span class="label">Token:</span> ${{escHtml(d.tok)}}<br>` +
        `<span class="label">Original:</span> ${{escHtml(d.orig)}}<br>` +
        `<span class="label">Chars:</span> ${{d.chars}}<br>` +
        `<span class="label">Bytes:</span> ${{d.bytes}}`;
      if (HAS_INFERENCE && parseInt(d.rank) > 0) {{
        html += `<br><span class="label">Rank:</span> ${{d.rank}}` +
                `<br><span class="label">Probability:</span> ${{parseFloat(d.prob).toFixed(6)}}`;
      }} else if (HAS_INFERENCE && parseInt(d.rank) === 0) {{
        html += `<br><span class="label">(first token – no prediction)</span>`;
      }}
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      pos(ev);
    }});
    el.addEventListener('mousemove', pos);
    el.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
  }});
}}

function renderLangRow(label, tokens, colours) {{
  let html = `<div class="lang-row"><span class="lang-label">${{label}}</span><span class="token-row">`;
  tokens.forEach((t, j) => {{
    let display = t.orig.replace(/\\n/g,'\\\\n').replace(/\\t/g,'\\\\t');
    if (display === ' ') display = '\\u2423';
    if (display === '') display = '\\u2205';
    const bg = colours[j], fg = fgFor(bg);
    html += `<span class="tok" style="background:${{bg}};color:${{fg}}" ` +
      `data-tid="${{t.id}}" data-tok="${{escHtml(t.tok)}}" data-orig="${{escHtml(t.orig)}}" ` +
      `data-chars="${{t.chars}}" data-bytes="${{t.bytes}}" ` +
      `data-rank="${{t.rank}}" data-prob="${{t.prob}}">${{escHtml(display)}}</span>`;
  }});
  html += '</span></div>';
  return html;
}}

function pos(ev) {{
  tooltip.style.left = (ev.clientX + 12) + 'px';
  tooltip.style.top = (ev.clientY + 12) + 'px';
}}

document.getElementById('colourMode').addEventListener('change', renderPairs);
renderSummary();
renderPairs();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test Gemma 270M tokenization on EN-KO translation pairs with heatmaps"
    )
    p.add_argument(
        "--model", type=str, default="google/gemma-3-270m",
        help="HuggingFace model for tokenizer (and model if --inference)",
    )
    p.add_argument(
        "--dataset", type=str, default="Helsinki-NLP/opus-100",
        help="HuggingFace translation dataset",
    )
    p.add_argument(
        "--dataset_config", type=str, default="en-ko",
        help="Dataset configuration / language pair",
    )
    p.add_argument("--num_pairs", type=int, default=5, help="Number of sentence pairs to process")
    p.add_argument(
        "--inference", action="store_true",
        help="Load the full model and compute per-token rank/probability",
    )
    p.add_argument(
        "--colour_mode",
        choices=["byte_length", "char_length", "token_id", "rank", "probability", "minmax"],
        default="byte_length",
        help="Colour mode for terminal output",
    )
    p.add_argument(
        "--rank_red", type=int, default=100,
        help="Rank value treated as fully red in rank colour mode",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (cpu, cuda, cuda:0, etc.)",
    )
    p.add_argument("--html", type=str, default=None, help="Path for interactive HTML output")
    p.add_argument("--output_file", type=str, default=None, help="Path to save terminal ANSI output")
    return p.parse_args()


def main():
    args = parse_args()

    if args.colour_mode in ("rank", "probability", "minmax") and not args.inference:
        print(f"Colour mode '{args.colour_mode}' requires --inference flag", file=sys.stderr)
        sys.exit(1)

    # Load data
    pairs = fetch_en_ko_pairs(args.num_pairs, args.dataset, args.dataset_config)
    print(f"Loaded {len(pairs)} EN-KO pairs")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tokenize all pairs
    en_all = [tokenize_text(en, tokenizer) for en, _ in pairs]
    ko_all = [tokenize_text(ko, tokenizer) for _, ko in pairs]

    # Inference
    en_inf = None
    ko_inf = None
    if args.inference:
        from transformers import AutoModelForCausalLM
        import torch

        print(f"Loading model: {args.model} on {args.device}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        ).to(args.device).eval()

        print("Running inference on EN texts...")
        en_inf = run_inference_batch(en_all, model, args.device)
        print("Running inference on KO texts...")
        ko_inf = run_inference_batch(ko_all, model, args.device)

    # Terminal output
    ansi = render_terminal_heatmap(
        pairs, en_all, ko_all,
        colour_mode=args.colour_mode,
        en_inf=en_inf, ko_inf=ko_inf,
        rank_red=args.rank_red,
    )
    print(ansi)

    if args.output_file:
        Path(args.output_file).write_text(ansi, "utf-8", errors="replace")
        print(f"Saved terminal output -> {args.output_file}")

    # HTML output
    if args.html:
        html_content = render_html_heatmap(
            pairs, en_all, ko_all,
            en_inf=en_inf, ko_inf=ko_inf,
            rank_red=args.rank_red,
        )
        Path(args.html).write_text(html_content, "utf-8")
        print(f"Saved interactive HTML -> {args.html}")


if __name__ == "__main__":
    main()

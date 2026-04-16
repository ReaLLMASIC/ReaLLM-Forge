# gemma_tokenization_highlighter.py
"""Visualise how the Gemma 270M tokenizer segments text.

Produces both a **terminal** (ANSI) view and an **interactive HTML** heatmap
that colour each token by its length in characters, byte-length, token-id,
or — with ``--inference`` — by model prediction **rank**, **probability**,
or **minmax**-normalised probability.

Inference modes (require loading the full model):

* **rank**        – green = rank 1 (model's top prediction), red = rank ≥ ``--rank_red``
* **probability** – green = 1.0 softmax confidence, red = 0.0
* **minmax**      – red = lowest probability in the sequence, green = highest

Works standalone with ``transformers`` and (optionally) ``rich``.

Usage
-----
```bash
# tokenization-only (fast, no model weights needed)
python gemma_tokenization_highlighter.py --text "Hello, world! 안녕하세요!"

# with inference heatmap
python gemma_tokenization_highlighter.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --inference --colour_mode probability \
    --html tokenization.html

# rank mode with custom ceiling
python gemma_tokenization_highlighter.py \
    --text "Hello, world!" --inference \
    --colour_mode rank --rank_red 50 \
    --html rank_view.html
```
"""
from __future__ import annotations

import argparse
import colorsys
import html as html_lib
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------

def _hue_to_rgb_hex(hue: float, saturation: float = 0.65, lightness: float = 0.50) -> str:
    """Convert HSL hue (0-1) to an #RRGGBB hex string."""
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _token_id_colour(token_id: int, vocab_size: int) -> str:
    """Deterministic colour based on token ID (golden-ratio hashing)."""
    golden = 0.618033988749895
    hue = (token_id * golden) % 1.0
    return _hue_to_rgb_hex(hue)


def _scalar_colour(value: float, lo: float, hi: float) -> str:
    """Red-to-green gradient for a scalar value between lo..hi."""
    if hi - lo < 1e-9:
        norm = 0.5
    else:
        norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


def _rg_gradient(norm: float) -> str:
    """Red (0) -> Green (1) absolute gradient."""
    norm = max(0.0, min(1.0, norm))
    r = int((1 - norm) * 255)
    g = int(norm * 255)
    return f"#{r:02x}{g:02x}00"


# ---------------------------------------------------------------------------
# tokenize and build spans
# ---------------------------------------------------------------------------

def tokenize_to_spans(
    text: str,
    tokenizer,
) -> List[Tuple[int, str, str]]:
    """Return list of (token_id, token_string, original_text_segment)."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    spans = []
    for tid, (start, end) in zip(ids, offsets):
        tok_str = tokenizer.convert_ids_to_tokens(tid)
        original = text[start:end]
        spans.append((tid, tok_str, original))
    return spans


# ---------------------------------------------------------------------------
# inference – compute rank and probability for each token
# ---------------------------------------------------------------------------

def run_inference(
    token_ids: List[int],
    model,
    tokenizer,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """For each token at position i (i>=1), compute rank and probability.

    Returns a list of dicts with keys: rank, probability.
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
        logits = outputs.logits  # (1, seq_len, vocab_size)

    logits = logits.squeeze(0)  # (seq_len, vocab_size)

    for i in range(1, len(token_ids)):
        # logits[i-1] predicts token at position i
        step_logits = logits[i - 1]
        probs = F.softmax(step_logits, dim=-1)
        target_id = token_ids[i]
        prob = probs[target_id].item()
        rank = int((step_logits > step_logits[target_id]).sum().item()) + 1
        results.append({"rank": rank, "probability": prob})

    return results


# ---------------------------------------------------------------------------
# colour mode implementations
# ---------------------------------------------------------------------------

COLOUR_MODES = ("token_id", "char_length", "byte_length", "rank", "probability", "minmax")
INFERENCE_MODES = {"rank", "probability", "minmax"}


def get_colours(
    spans: List[Tuple[int, str, str]],
    mode: str,
    vocab_size: int,
    inference_results: Optional[List[Dict[str, float]]] = None,
    rank_red: int = 100,
) -> List[str]:
    """Return one hex colour string per span."""
    if mode == "token_id":
        return [_token_id_colour(tid, vocab_size) for tid, _, _ in spans]

    if mode == "char_length":
        lengths = [len(orig) for _, _, orig in spans]
        lo, hi = (min(lengths), max(lengths)) if lengths else (0, 1)
        return [_scalar_colour(l, lo, hi) for l in lengths]

    if mode == "byte_length":
        lengths = [len(orig.encode("utf-8")) for _, _, orig in spans]
        lo, hi = (min(lengths), max(lengths)) if lengths else (0, 1)
        return [_scalar_colour(l, lo, hi) for l in lengths]

    # Inference-based modes
    if inference_results is None:
        return ["#888888"] * len(spans)

    if mode == "rank":
        colours = []
        for res in inference_results:
            rank = res["rank"]
            if rank == 0:  # first token, no prediction
                colours.append("#888888")
            else:
                # rank 1 = green (norm=1), rank >= rank_red = red (norm=0)
                norm = 1.0 - (min(rank, rank_red) - 1) / max(rank_red - 1, 1)
                colours.append(_rg_gradient(norm))
        return colours

    if mode == "probability":
        # Absolute: 1.0 = green, 0.0 = red
        colours = []
        for res in inference_results:
            if res["rank"] == 0:
                colours.append("#888888")
            else:
                colours.append(_rg_gradient(res["probability"]))
        return colours

    if mode == "minmax":
        # Relative: lowest prob in sequence = red, highest = green
        probs = [res["probability"] for res in inference_results if res["rank"] > 0]
        if not probs:
            return ["#888888"] * len(spans)
        lo, hi = min(probs), max(probs)
        colours = []
        for res in inference_results:
            if res["rank"] == 0:
                colours.append("#888888")
            else:
                colours.append(_scalar_colour(res["probability"], lo, hi))
        return colours

    return ["#888888"] * len(spans)


# ---------------------------------------------------------------------------
# terminal output (Rich)
# ---------------------------------------------------------------------------

def render_terminal(
    spans: List[Tuple[int, str, str]],
    colours: List[str],
    show_boundaries: bool = True,
    inference_results: Optional[List[Dict[str, float]]] = None,
) -> str:
    """Render coloured tokens to the terminal via Rich."""
    try:
        from rich.console import Console
        from rich.text import Text
    except ImportError:
        parts = []
        for (tid, tok_str, orig), col in zip(spans, colours):
            display = repr(orig)[1:-1] if show_boundaries else orig
            parts.append(f"\033[1m{display}\033[0m")
        return "".join(parts)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    text = Text()
    for (tid, tok_str, orig), col in zip(spans, colours):
        display = orig.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if show_boundaries:
            display = f"|{display}"
        text.append(display, style=f"bold {col}")
    if show_boundaries:
        text.append("|", style="dim")
    console.print(text)

    # Summary
    console.print()
    console.print(f"[cyan]Total tokens:[/cyan] {len(spans)}")
    console.print(f"[cyan]Total chars:[/cyan]  {sum(len(o) for _, _, o in spans)}")
    console.print(f"[cyan]Total bytes:[/cyan]  {sum(len(o.encode('utf-8')) for _, _, o in spans)}")

    # Inference summary
    if inference_results is not None:
        probs = [r["probability"] for r in inference_results if r["rank"] > 0]
        ranks = [r["rank"] for r in inference_results if r["rank"] > 0]
        if probs:
            import statistics
            avg_prob = statistics.mean(probs)
            avg_rank = statistics.mean(ranks)
            median_rank = statistics.median(ranks)
            console.print(f"[cyan]Avg probability:[/cyan] {avg_prob:.4f}")
            console.print(f"[cyan]Avg rank:[/cyan]        {avg_rank:.1f}")
            console.print(f"[cyan]Median rank:[/cyan]     {median_rank:.0f}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def render_html(
    text: str,
    spans: List[Tuple[int, str, str]],
    colours: List[str],
    title: str = "Gemma 270M Tokenization",
    inference_results: Optional[List[Dict[str, float]]] = None,
    rank_red: int = 100,
) -> str:
    """Build a self-contained interactive HTML page."""
    has_inference = inference_results is not None

    token_data_js = []
    for i, ((tid, tok_str, orig), col) in enumerate(zip(spans, colours)):
        inf = inference_results[i] if has_inference else {"rank": -1, "probability": -1}
        token_data_js.append(
            f'{{id:{tid},tok:{_js_str(tok_str)},orig:{_js_str(orig)},'
            f'chars:{len(orig)},bytes:{len(orig.encode("utf-8"))},'
            f'col:"{col}",rank:{inf["rank"]},prob:{inf["probability"]:.6f}}}'
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
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ margin-bottom: 10px; font-size: 1.4em; color: #e0e0ff; }}
  .stats {{ color: #aaa; margin-bottom: 16px; font-size: 0.9em; }}
  .controls {{ margin-bottom: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85em; color: #ccc; }}
  .controls select, .controls input {{ background: #2a2a4a; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }}
  #token-container {{ line-height: 2.2; padding: 16px; background: #16213e; border-radius: 8px; overflow-x: auto; }}
  .tok {{
    display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px;
    cursor: pointer; font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.95em;
    transition: transform 0.1s, box-shadow 0.1s;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .tok:hover {{ transform: scale(1.1); box-shadow: 0 0 8px rgba(255,255,255,0.3); z-index: 10; position: relative; }}
  #tooltip {{
    display: none; position: fixed; background: #0f3460; color: #fff;
    padding: 10px 14px; border-radius: 6px; font-size: 0.85em; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 400px;
    font-family: 'Fira Code', monospace; line-height: 1.6;
  }}
  #tooltip .label {{ color: #7ec8e3; }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<div class="stats" id="stats"></div>
<div class="controls">
  <label>Colour by:
    <select id="colourMode">
      <option value="token_id">Token ID (hashed)</option>
      <option value="char_length">Char length</option>
      <option value="byte_length">Byte length</option>{inference_options}
    </select>
  </label>
  <label>Show boundaries:
    <input type="checkbox" id="showBounds" checked>
  </label>
</div>
<div id="token-container"></div>
<div id="tooltip"></div>

<script>
const tokens = [{",".join(token_data_js)}];
const RANK_RED = {rank_red};
const HAS_INFERENCE = {'true' if has_inference else 'false'};
const container = document.getElementById('token-container');
const tooltip = document.getElementById('tooltip');
const stats = document.getElementById('stats');

let statsText = `Total tokens: ${{tokens.length}} | Total chars: ${{tokens.reduce((s,t)=>s+t.chars,0)}} | Total bytes: ${{tokens.reduce((s,t)=>s+t.bytes,0)}}`;
if (HAS_INFERENCE) {{
  const scored = tokens.filter(t => t.rank > 0);
  if (scored.length > 0) {{
    const avgP = (scored.reduce((s,t)=>s+t.prob,0) / scored.length).toFixed(4);
    const avgR = (scored.reduce((s,t)=>s+t.rank,0) / scored.length).toFixed(1);
    statsText += ` | Avg prob: ${{avgP}} | Avg rank: ${{avgR}}`;
  }}
}}
stats.textContent = statsText;

const golden = 0.618033988749895;
function hueToHex(h, s=0.65, l=0.5) {{
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

function recolour(mode) {{
  const spans = container.querySelectorAll('.tok');
  if (mode === 'token_id') {{
    spans.forEach((el, i) => {{
      const c = hueToHex((tokens[i].id * golden) % 1.0);
      el.style.background = c; el.style.color = fgFor(c);
    }});
  }} else if (mode === 'char_length' || mode === 'byte_length') {{
    const key = mode === 'char_length' ? 'chars' : 'bytes';
    const vals = tokens.map(t => t[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    spans.forEach((el, i) => {{
      const c = scalarCol(vals[i], lo, hi);
      el.style.background = c; el.style.color = fgFor(c);
    }});
  }} else if (mode === 'rank') {{
    spans.forEach((el, i) => {{
      const t = tokens[i];
      let c;
      if (t.rank === 0) {{ c = '#888888'; }}
      else {{ c = rgGrad(1.0 - (Math.min(t.rank, RANK_RED) - 1) / Math.max(RANK_RED - 1, 1)); }}
      el.style.background = c; el.style.color = fgFor(c);
    }});
  }} else if (mode === 'probability') {{
    spans.forEach((el, i) => {{
      const t = tokens[i];
      const c = t.rank === 0 ? '#888888' : rgGrad(t.prob);
      el.style.background = c; el.style.color = fgFor(c);
    }});
  }} else if (mode === 'minmax') {{
    const scored = tokens.filter(t => t.rank > 0);
    const probs = scored.map(t => t.prob);
    const lo = Math.min(...probs), hi = Math.max(...probs);
    spans.forEach((el, i) => {{
      const t = tokens[i];
      const c = t.rank === 0 ? '#888888' : scalarCol(t.prob, lo, hi);
      el.style.background = c; el.style.color = fgFor(c);
    }});
  }}
}}

function render() {{
  container.innerHTML = '';
  tokens.forEach((t, i) => {{
    const el = document.createElement('span');
    el.className = 'tok';
    let display = t.orig.replace(/\\n/g, '\\\\n').replace(/\\t/g, '\\\\t').replace(/\\r/g, '\\\\r');
    if (display === ' ') display = '\\u2423';
    if (display === '') display = '\\u2205';
    el.textContent = display;
    el.style.background = t.col;
    el.style.color = fgFor(t.col);
    el.addEventListener('mouseenter', (ev) => {{
      let html =
        `<span class="label">Token ID:</span> ${{t.id}}<br>` +
        `<span class="label">Token:</span> ${{escHtml(t.tok)}}<br>` +
        `<span class="label">Original:</span> ${{escHtml(t.orig)}}<br>` +
        `<span class="label">Chars:</span> ${{t.chars}}<br>` +
        `<span class="label">Bytes:</span> ${{t.bytes}}`;
      if (HAS_INFERENCE && t.rank > 0) {{
        html += `<br><span class="label">Rank:</span> ${{t.rank}}` +
                `<br><span class="label">Probability:</span> ${{t.prob.toFixed(6)}}`;
      }} else if (HAS_INFERENCE && t.rank === 0) {{
        html += `<br><span class="label">(first token – no prediction)</span>`;
      }}
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      positionTooltip(ev);
    }});
    el.addEventListener('mousemove', positionTooltip);
    el.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});
    container.appendChild(el);
  }});
  recolour(document.getElementById('colourMode').value);
}}

function positionTooltip(ev) {{
  tooltip.style.left = (ev.clientX + 12) + 'px';
  tooltip.style.top = (ev.clientY + 12) + 'px';
}}

function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

document.getElementById('colourMode').addEventListener('change', (e) => recolour(e.target.value));
document.getElementById('showBounds').addEventListener('change', render);
render();
</script>
</body>
</html>"""


def _js_str(s: str) -> str:
    """Escape a Python string for safe embedding in JS."""
    return (
        '"'
        + s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + '"'
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise Gemma 270M tokenization with colour highlighting"
    )
    p.add_argument("--text", type=str, default=None, help="Text to tokenize")
    p.add_argument("--input_file", type=str, default=None, help="Read text from a file")
    p.add_argument(
        "--model", type=str, default="google/gemma-3-270m",
        help="HuggingFace model name for tokenizer (and model if --inference)",
    )
    p.add_argument(
        "--colour_mode", choices=COLOUR_MODES, default="token_id",
        help="How to assign colours to tokens",
    )
    p.add_argument(
        "--inference", action="store_true",
        help="Load the full model and compute per-token rank/probability",
    )
    p.add_argument(
        "--rank_red", type=int, default=100,
        help="Rank value treated as fully red in rank colour mode",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (cpu, cuda, cuda:0, etc.)",
    )
    p.add_argument("--html", type=str, default=None, help="Path to write interactive HTML output")
    p.add_argument("--output_file", type=str, default=None, help="Path to save terminal ANSI output")
    p.add_argument(
        "--no_boundaries", action="store_true",
        help="Hide token boundary markers in terminal output",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input_file is None:
        print("Provide --text or --input_file", file=sys.stderr)
        sys.exit(1)

    if args.colour_mode in INFERENCE_MODES and not args.inference:
        print(f"Colour mode '{args.colour_mode}' requires --inference flag", file=sys.stderr)
        sys.exit(1)

    text = args.text
    if args.input_file:
        text = Path(args.input_file).read_text("utf-8")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    spans = tokenize_to_spans(text, tokenizer)
    token_ids = [tid for tid, _, _ in spans]

    # Inference
    inference_results = None
    if args.inference:
        from transformers import AutoModelForCausalLM
        import torch

        print(f"Loading model: {args.model} on {args.device}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        ).to(args.device).eval()

        print(f"Running inference on {len(token_ids)} tokens...")
        inference_results = run_inference(token_ids, model, tokenizer, args.device)

    colours = get_colours(
        spans, args.colour_mode, tokenizer.vocab_size,
        inference_results=inference_results, rank_red=args.rank_red,
    )

    # Terminal output
    ansi = render_terminal(
        spans, colours,
        show_boundaries=not args.no_boundaries,
        inference_results=inference_results,
    )
    print(ansi)

    if args.output_file:
        Path(args.output_file).write_text(ansi, "utf-8", errors="replace")
        print(f"Saved terminal output -> {args.output_file}")

    # HTML output
    if args.html:
        html_content = render_html(
            text, spans, colours,
            inference_results=inference_results,
            rank_red=args.rank_red,
        )
        Path(args.html).write_text(html_content, "utf-8")
        print(f"Saved interactive HTML -> {args.html}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
spm_vocab_freq_dashboard.py

Single-script, self-contained HTML dashboard (Plotly + vanilla JS) that shows:

LEFT:
  - SentencePiece vocab tokens + total frequency across *all* text files in a directory
  - searchable dropdown to pick a token
  - click-to-select from the token table

RIGHT (top):
  - per-file counts for the selected token (bar chart)

RIGHT (bottom):
  - square similarity heatmap clustering text files by similarity across high-frequency vocab
    (cosine similarity over TF-IDF on top vocab tokens)

Why we require a .model:
  - SentencePiece tokenization is not substring matching; to get true token frequencies,
    we MUST encode text using the SentencePiece model.

Defaults:
  --vocab trained_spm_model.vocab
  --model inferred by replacing ".vocab" -> ".model" if not provided
  --dir   required

Output:
  vocab_freq_dashboard.html (or --out)

Example:
  python3 spm_vocab_freq_dashboard.py --dir ./text --vocab trained_spm_model.vocab --heatmap
  python3 spm_vocab_freq_dashboard.py --dir ./text --heatmap-top-k 500 --recursive
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sentencepiece as spm

# For heatmap similarity: NumPy required (SciPy not needed)
try:
    import numpy as np
except Exception as e:
    np = None
    _NUMPY_IMPORT_ERROR = e


def infer_model_path_from_vocab(vocab_path: Path) -> Path:
    if vocab_path.suffix.lower() == ".vocab":
        return vocab_path.with_suffix(".model")
    return Path(str(vocab_path) + ".model")


def iter_text_files(root: Path, recursive: bool, suffixes: Tuple[str, ...]) -> List[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    files = []
    for p in it:
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    files.sort()
    return files


def count_tokens_in_file(sp: spm.SentencePieceProcessor, path: Path) -> Counter:
    """
    Streaming-ish: encode line by line to avoid loading huge files into memory.
    """
    c = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ids = sp.encode(line, out_type=int)
            c.update(ids)
    return c


def human_token(tok: str) -> str:
    return tok.replace("\t", " ").replace("\n", "\\n")


def _build_tfidf_matrix(
    file_names: List[str],
    per_file: Dict[str, Counter],
    token_ids: List[int],
) -> "np.ndarray":
    """
    docs x tokens TF-IDF, L2-normalized per doc
    """
    assert np is not None
    n_docs = len(file_names)
    n_tok = len(token_ids)
    if n_docs == 0 or n_tok == 0:
        return np.zeros((n_docs, n_tok), dtype=np.float32)

    tok_to_col = {tid: j for j, tid in enumerate(token_ids)}

    # document frequency
    df = np.zeros((n_tok,), dtype=np.int32)
    for fn in file_names:
        c = per_file[fn]
        for tid in c.keys():
            j = tok_to_col.get(tid)
            if j is not None:
                df[j] += 1

    # smooth idf
    idf = np.log((n_docs + 1.0) / (df.astype(np.float32) + 1.0)) + 1.0

    X = np.zeros((n_docs, n_tok), dtype=np.float32)
    for i, fn in enumerate(file_names):
        c = per_file[fn]
        row = X[i]
        for tid, cnt in c.items():
            j = tok_to_col.get(tid)
            if j is not None:
                row[j] = float(cnt)

        row *= idf

        # L2 normalize
        norm = float(np.linalg.norm(row))
        if norm > 0:
            row /= norm

    return X


def _cosine_similarity_matrix(X: "np.ndarray") -> "np.ndarray":
    """
    X assumed rows L2-normalized; cosine similarity = X @ X.T
    """
    assert np is not None
    if X.size == 0:
        return np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    S = X @ X.T
    # numerical guard
    S = np.clip(S, -1.0, 1.0).astype(np.float32)
    # make diagonal exactly 1
    n = S.shape[0]
    for i in range(n):
        S[i, i] = 1.0
    return S


def _order_by_simple_clustering(S: "np.ndarray") -> List[int]:
    """
    Optional: reorder files so similar ones are near each other, without SciPy.
    Greedy "nearest neighbor chain" heuristic:
      - start from most "central" (max average similarity)
      - repeatedly append most similar unused item to the last
    """
    assert np is not None
    n = S.shape[0]
    if n <= 2:
        return list(range(n))

    avg = S.mean(axis=1)
    start = int(np.argmax(avg))
    order = [start]
    used = set(order)

    while len(order) < n:
        last = order[-1]
        # pick unused with max similarity to last
        best_j = None
        best_val = -1e9
        for j in range(n):
            if j in used:
                continue
            v = float(S[last, j])
            if v > best_val:
                best_val = v
                best_j = j
        order.append(int(best_j))
        used.add(int(best_j))

    return order


def _build_heatmap_payload(
    file_names: List[str],
    per_file: Dict[str, Counter],
    token_ids_for_heatmap: List[int],
    reorder: bool,
) -> Dict:
    """
    Returns plotly-ready payload for similarity heatmap.
    """
    if np is None:
        raise RuntimeError(
            "NumPy is required for heatmap mode.\n"
            f"Import error: {_NUMPY_IMPORT_ERROR!r}\n"
            "Install: python3 -m pip install numpy"
        )

    if len(file_names) < 2:
        return {"ok": False, "reason": "Need at least 2 files to build a similarity heatmap."}

    X = _build_tfidf_matrix(file_names, per_file, token_ids_for_heatmap)
    S = _cosine_similarity_matrix(X)

    idx = list(range(len(file_names)))
    if reorder:
        idx = _order_by_simple_clustering(S)

    labels = [file_names[i] for i in idx]
    S2 = S[np.ix_(idx, idx)]

    # convert to nested lists for JSON
    z = S2.tolist()

    traces = [{
        "type": "heatmap",
        "z": z,
        "x": labels,
        "y": labels,
        "zmin": 0.0,
        "zmax": 1.0,
        "hovertemplate": "x=%{x}<br>y=%{y}<br>cosine=%{z:.3f}<extra></extra>",
        # no explicit colorscale specified (Plotly default) to match your earlier “no custom colors” vibe
    }]

    layout = {
        "margin": {"l": 120, "r": 20, "t": 40, "b": 120},
        "title": "File similarity heatmap (TF-IDF on high-freq SPM vocab, cosine similarity)",
        "xaxis": {"tickangle": 35, "automargin": True},
        "yaxis": {"automargin": True},
    }

    return {"ok": True, "traces": traces, "layout": layout}


def build_html(
    title: str,
    token_rows: List[Dict],
    per_file_counts: Dict[str, Dict[str, int]],
    file_order: List[str],
    default_token_id: int,
    heatmap_payload: Optional[Dict],
    out_path: Path,
) -> None:
    payload = {
        "title": title,
        "tokens": token_rows,
        "per_file": per_file_counts,
        "files": file_order,
        "default_token_id": default_token_id,
        "heatmap": heatmap_payload,
    }

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Noto Sans", Arial, sans-serif;
      margin: 0; padding: 0;
    }}
    header {{
      padding: 12px 16px;
      border-bottom: 1px solid #ddd;
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    header h1 {{
      font-size: 16px;
      margin: 0;
      flex: 1;
    }}
    .container {{
      display: flex;
      height: calc(100vh - 58px);
      gap: 10px;
      padding: 10px;
      box-sizing: border-box;
    }}
    .panel {{
      flex: 1;
      min-width: 380px;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 10px;
      box-sizing: border-box;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .panel h2 {{
      font-size: 14px;
      margin: 0 0 8px 0;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .controls label {{
      font-size: 12px;
      color: #333;
    }}
    select, input {{
      font-size: 12px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid #ccc;
      outline: none;
    }}
    #tableDiv {{
      flex: 1;
      min-height: 200px;
    }}
    .rightCharts {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex: 1;
      overflow: hidden;
    }}
    #barDiv {{
      flex: 1;
      min-height: 200px;
    }}
    #heatDiv {{
      flex: 1;
      min-height: 260px;
    }}
    .note {{
      font-size: 12px;
      color: #555;
      padding: 8px 0 0 0;
    }}
    .muted {{
      color: #777;
    }}
  </style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="muted">Pick a token to update per-file counts; heatmap shows file similarity via high-frequency vocab.</span>
</header>

<div class="container">
  <div class="panel">
    <h2>Vocab + total frequency (directory aggregate)</h2>
    <div class="controls">
      <label for="tokenSelect">Token:</label>
      <select id="tokenSelect"></select>
      <label for="searchBox">Search:</label>
      <input id="searchBox" type="text" placeholder="type to filter tokens..." />
    </div>
    <div id="tableDiv"></div>
    <div class="note">
      Showing <b>top {len(token_rows)}</b> tokens by frequency.
    </div>
  </div>

  <div class="panel">
    <h2 id="rightTitle">Per-file counts + similarity heatmap</h2>
    <div class="rightCharts">
      <div id="barDiv"></div>
      <div id="heatDiv"></div>
    </div>
    <div class="note">
      Heatmap uses <b>TF-IDF</b> over high-frequency SentencePiece tokens and <b>cosine similarity</b>.
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};

function fmtTokenRow(t) {{
  let s = t.token;
  if (s.length > 60) s = s.slice(0, 57) + "…";
  return `${{t.id}}: ${{s}} (${{t.count}})`;
}}

function buildSelectOptions(tokens) {{
  const sel = document.getElementById("tokenSelect");
  sel.innerHTML = "";
  for (const t of tokens) {{
    const opt = document.createElement("option");
    opt.value = String(t.id);
    opt.textContent = fmtTokenRow(t);
    sel.appendChild(opt);
  }}
}}

function filterTokens(tokens, q) {{
  if (!q) return tokens;
  q = q.toLowerCase();
  return tokens.filter(t => String(t.id).includes(q) || (t.token || "").toLowerCase().includes(q));
}}

function renderTable(tokens) {{
  const ids = tokens.map(t => t.id);
  const toks = tokens.map(t => t.token);
  const counts = tokens.map(t => t.count);

  const tableData = [{{
    type: "table",
    header: {{
      values: ["<b>ID</b>", "<b>Token</b>", "<b>Total Count</b>"],
      align: ["right", "left", "right"],
    }},
    cells: {{
      values: [ids, toks, counts],
      align: ["right", "left", "right"],
      height: 22
    }}
  }}];

  Plotly.newPlot("tableDiv", tableData, {{
    margin: {{l: 10, r: 10, t: 10, b: 10}},
  }}, {{displayModeBar: false}});

  const tableDiv = document.getElementById("tableDiv");
  tableDiv.on("plotly_click", (ev) => {{
    try {{
      const row = ev.points[0].pointNumber;
      const tok = tokens[row];
      if (tok) {{
        selectToken(String(tok.id), true);
      }}
    }} catch (e) {{}}
  }});
}}

function renderBar(tokenId) {{
  const tok = DATA.tokens.find(t => String(t.id) === String(tokenId));
  const name = tok ? tok.token : `(id=${{tokenId}})`;

  const per = DATA.per_file[String(tokenId)] || {{}};
  const xs = DATA.files.slice();
  const ys = xs.map(fn => (per[fn] || 0));

  Plotly.newPlot("barDiv", [{{
    type: "bar",
    x: xs,
    y: ys
  }}], {{
    margin: {{l: 50, r: 10, t: 30, b: 120}},
    xaxis: {{ tickangle: 35, automargin: true }},
    yaxis: {{ title: "Count" }},
    title: `Token: ${{name}} (id=${{tokenId}})`
  }}, {{displayModeBar: true}});
}}

function renderHeatmap() {{
  const h = DATA.heatmap;
  const div = document.getElementById("heatDiv");

  if (!h) {{
    Plotly.newPlot(div, [], {{
      margin: {{l: 20, r: 10, t: 30, b: 30}},
      title: "Similarity heatmap: not computed",
      annotations: [{{
        text: "No heatmap payload present.",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false
      }}]
    }}, {{displayModeBar: false}});
    return;
  }}

  if (!h.ok) {{
    Plotly.newPlot(div, [], {{
      margin: {{l: 20, r: 10, t: 30, b: 30}},
      title: "Similarity heatmap: unavailable",
      annotations: [{{
        text: h.reason || "Unavailable",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false
      }}]
    }}, {{displayModeBar: false}});
    return;
  }}

  Plotly.newPlot(div, h.traces, h.layout, {{displayModeBar: true}});
}}

function selectToken(tokenId, updateSelect) {{
  if (updateSelect) {{
    document.getElementById("tokenSelect").value = String(tokenId);
  }}
  renderBar(tokenId);
}}

function init() {{
  buildSelectOptions(DATA.tokens);

  const sel = document.getElementById("tokenSelect");
  sel.value = String(DATA.default_token_id);

  renderTable(DATA.tokens);
  renderBar(DATA.default_token_id);
  renderHeatmap();

  sel.addEventListener("change", (e) => {{
    selectToken(e.target.value, false);
  }});

  const search = document.getElementById("searchBox");
  search.addEventListener("input", (e) => {{
    const q = e.target.value || "";
    const filtered = filterTokens(DATA.tokens, q);

    const cur = document.getElementById("tokenSelect").value;
    buildSelectOptions(filtered);

    const hasCur = filtered.some(t => String(t.id) === String(cur));
    const newId = hasCur ? cur : (filtered.length ? String(filtered[0].id) : String(DATA.default_token_id));
    document.getElementById("tokenSelect").value = newId;

    renderTable(filtered);
    selectToken(newId, false);
  }});
}}

init();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="trained_spm_model.vocab",
                    help="SentencePiece vocab file (default: trained_spm_model.vocab). Used to infer .model if --model not given.")
    ap.add_argument("--model", default=None,
                    help="SentencePiece model file (.model). If omitted, inferred from --vocab by replacing .vocab -> .model.")
    ap.add_argument("--dir", required=True,
                    help="Directory of text files to scan.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories (default: false).")
    ap.add_argument("--suffixes", default=".txt",
                    help="Comma-separated suffixes to include (default: .txt). Example: .txt,.md")

    ap.add_argument("--top-k", type=int, default=1500,
                    help="Embed only top-K tokens by total frequency into the HTML (default: 1500).")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Only consider tokens with total count >= this (default: 1).")
    ap.add_argument("--out", default="vocab_freq_dashboard.html",
                    help="Output HTML path (default: vocab_freq_dashboard.html)")

    # NEW: heatmap controls (no dendro mode; just heatmap)
    ap.add_argument("--heatmap", action="store_true",
                    help="Compute and embed a file similarity heatmap (requires numpy).")
    ap.add_argument("--heatmap-top-k", type=int, default=300,
                    help="Use top-K frequent tokens (from the directory) as features for TF-IDF similarity (default: 300).")
    ap.add_argument("--heatmap-reorder", action="store_true",
                    help="Reorder files to group similar ones (simple greedy heuristic, no SciPy).")

    args = ap.parse_args()

    vocab_path = Path(args.vocab)
    model_path = Path(args.model) if args.model else infer_model_path_from_vocab(vocab_path)
    root = Path(args.dir)
    out = Path(args.out)

    if not model_path.exists():
        raise SystemExit(f"SentencePiece model not found: {model_path} (pass --model or ensure it matches --vocab)")
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    suffixes = tuple(s.strip().lower() for s in args.suffixes.split(",") if s.strip())
    files = iter_text_files(root, recursive=args.recursive, suffixes=suffixes)
    if not files:
        raise SystemExit(f"No files found in {root} with suffixes={suffixes} (try --recursive or --suffixes)")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    print(f"[info] model: {model_path}")
    print(f"[info] vocab size: {sp.get_piece_size()}")
    print(f"[info] scanning {len(files)} files under: {root}")

    total = Counter()
    per_file: Dict[str, Counter] = {}
    file_names: List[str] = []

    for p in files:
        rel = str(p.relative_to(root))
        file_names.append(rel)
        c = count_tokens_in_file(sp, p)
        per_file[rel] = c
        total.update(c)

    # Token UI: top tokens by directory frequency
    items = [(tid, cnt) for tid, cnt in total.items() if cnt >= args.min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    if not items:
        raise SystemExit(f"No tokens met min-count={args.min_count} (unexpected).")

    top_items = items[: max(1, args.top_k)]

    token_rows: List[Dict] = []
    per_file_counts: Dict[str, Dict[str, int]] = {}

    for tid, cnt in top_items:
        tok = human_token(sp.id_to_piece(int(tid)))
        token_rows.append({"id": int(tid), "token": tok, "count": int(cnt)})

    for tid, _ in top_items:
        tid = int(tid)
        k = str(tid)
        per_file_counts[k] = {}
        for fn in file_names:
            v = per_file[fn].get(tid, 0)
            if v:
                per_file_counts[k][fn] = int(v)

    default_token_id = int(top_items[0][0])

    heatmap_payload: Optional[Dict] = None
    if args.heatmap:
        # Features for similarity
        feat_tok_ids = [int(tid) for tid, _ in items[: max(2, args.heatmap_top_k)]]
        print(f"[info] heatmap features: top {len(feat_tok_ids)} tokens (TF-IDF)")
        try:
            heatmap_payload = _build_heatmap_payload(
                file_names=file_names,
                per_file=per_file,
                token_ids_for_heatmap=feat_tok_ids,
                reorder=args.heatmap_reorder,
            )
        except Exception as e:
            heatmap_payload = {"ok": False, "reason": f"Failed to build heatmap: {e!r}"}
            print(f"[warn] heatmap failed: {e!r}")

    title = f"SentencePiece token frequency dashboard ({root.name})"
    build_html(
        title=title,
        token_rows=token_rows,
        per_file_counts=per_file_counts,
        file_order=file_names,
        default_token_id=default_token_id,
        heatmap_payload=heatmap_payload,
        out_path=out,
    )

    print(f"[done] wrote: {out}")
    print("[note] Uses Plotly CDN for interactivity; open the HTML in your browser.")
    if args.heatmap and (np is None):
        print("[note] Install heatmap deps: python3 -m pip install numpy")


if __name__ == "__main__":
    main()


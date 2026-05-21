#!/usr/bin/env python3
"""Interactive token-angle dashboard + island exporter for Gemma vocab embeddings."""
from __future__ import annotations

import argparse
import math
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank = self.rank
        parent = self.parent
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token angle dashboard + island exporter")
    p.add_argument("--model", default="google/gemma-3-270m")
    p.add_argument("--embedding-source", choices=["input", "lm_head"], default="input")
    p.add_argument("--tokens", required=True, help="Comma-separated token strings, e.g. '0,1,2'")
    p.add_argument("--angle-threshold-deg", type=float, default=70.0)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--max-vocab", type=int, default=-1)
    p.add_argument("--max-plot-vocab", type=int, default=20000)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--min-island-size", type=int, default=2)
    p.add_argument("--output-dir", default="./gemma_token_angle_dashboard")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _progress(processed: int, total: int, started: float) -> None:
    elapsed = max(time.time() - started, 1e-6)
    rate = processed / elapsed
    eta = (total - processed) / max(rate, 1e-9)
    print(f"\rBlocks: {processed}/{total} ({100*processed/max(total,1):5.1f}%) ETA {eta:7.1f}s", end="", flush=True)


def _resolve_token_ids(tokenizer: AutoTokenizer, token_texts: list[str]) -> list[int]:
    ids: list[int] = []
    for tok in token_texts:
        tok = tok.strip()
        token_id = tokenizer.convert_tokens_to_ids(tok)
        if token_id is None or token_id == tokenizer.unk_token_id:
            pieces = tokenizer(tok, add_special_tokens=False)["input_ids"]
            if not pieces:
                raise ValueError(f"Could not tokenize token text: {tok!r}")
            token_id = int(pieces[0])
        ids.append(int(token_id))
    return ids


def _build_islands(
    emb: torch.Tensor,
    cosine_threshold: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = emb.size(0)
    uf = UnionFind(n)
    degrees = np.zeros(n, dtype=np.int64)

    blocks = math.ceil(n / chunk_size)
    total_pairs = blocks * (blocks + 1) // 2
    done = 0
    started = time.time()

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        a = emb[i0:i1]
        for j0 in range(i0, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            b = emb[j0:j1]
            sims = torch.matmul(a, b.T)
            mask = sims >= cosine_threshold
            if i0 == j0:
                tri = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
                mask = mask & tri
            pairs = mask.nonzero(as_tuple=False)
            if pairs.numel() > 0:
                src = (pairs[:, 0] + i0).cpu().numpy().astype(np.int32)
                dst = (pairs[:, 1] + j0).cpu().numpy().astype(np.int32)
                degrees += np.bincount(src, minlength=n)
                degrees += np.bincount(dst, minlength=n)
                for s, d in zip(src, dst):
                    uf.union(int(s), int(d))
            done += 1
            if done % 10 == 0 or done == total_pairs:
                _progress(done, total_pairs, started)
    print()

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    _, component_ids = np.unique(roots, return_inverse=True)
    component_ids = component_ids.astype(np.int32)
    return component_ids, degrees


def _write_island_files(
    output_dir: Path,
    tokenizer: AutoTokenizer,
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
    degrees: np.ndarray,
    min_island_size: int,
) -> None:
    islands_dir = output_dir / "islands"
    islands_dir.mkdir(parents=True, exist_ok=True)

    members_by_component: dict[int, list[int]] = {}
    for token_id, component_id in enumerate(component_ids.tolist()):
        members_by_component.setdefault(component_id, []).append(token_id)

    for component_id, members in members_by_component.items():
        if len(members) < min_island_size:
            continue
        file_name = f"island_{component_id}_size{len(members)}_{uuid.uuid4().hex[:10]}.txt"
        path = islands_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            f.write(f"component_id={component_id}\n")
            f.write(f"component_size={len(members)}\n")
            f.write("token_id\ttoken\tdegree\n")
            for token_id in members:
                token = tokenizer.convert_ids_to_tokens(token_id).replace("\n", "\\n")
                f.write(f"{token_id}\t{token}\t{int(degrees[token_id])}\n")


def _write_selected_token_csvs(
    output_dir: Path,
    tokenizer: AutoTokenizer,
    emb: torch.Tensor,
    selected_token_ids: list[int],
    selected_token_texts: list[str],
    top_k: int,
) -> np.ndarray:
    sims = torch.matmul(emb[selected_token_ids], emb.T).cpu().numpy()
    sims = np.clip(sims, -1.0, 1.0)
    angles = np.degrees(np.arccos(sims))

    out_dir = output_dir / "selected_token_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, token_id in enumerate(selected_token_ids):
        order = np.argsort(angles[i])
        top = order[: max(1, min(top_k, len(order)))]
        csv_path = out_dir / f"token_{token_id}_{selected_token_texts[i].replace(' ', '_')}.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("rank,target_token_id,target_token,cosine,angle_deg\n")
            for rank, target_id in enumerate(top, start=1):
                tok = tokenizer.convert_ids_to_tokens(int(target_id)).replace("\n", "\\n")
                f.write(
                    f"{rank},{int(target_id)},{tok},{float(sims[i, target_id]):.8f},{float(angles[i, target_id]):.6f}\n"
                )
    return angles


def _write_plotly_dashboard(
    output_dir: Path,
    tokenizer: AutoTokenizer,
    selected_token_ids: list[int],
    selected_token_texts: list[str],
    angles: np.ndarray,
    max_plot_vocab: int,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_vocab = min(max_plot_vocab, angles.shape[1])
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Angle Distribution", "Token ID vs Angle"))

    for i, (tid, text) in enumerate(zip(selected_token_ids, selected_token_texts)):
        fig.add_trace(
            go.Histogram(x=angles[i], name=f"{text} (id={tid})", opacity=0.5, nbinsx=90),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=np.arange(plot_vocab),
                y=angles[i, :plot_vocab],
                mode="markers",
                marker={"size": 3},
                name=f"{text} (id={tid})",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        barmode="overlay",
        height=900,
        title="Selected Token Angle Dashboard",
        xaxis2_title="Token ID (truncated for plotting)",
        yaxis2_title="Angle (degrees)",
    )

    html_path = output_dir / "selected_token_dashboard.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    with (output_dir / "selected_tokens_resolved.txt").open("w", encoding="utf-8") as f:
        f.write("selected_token_text\tselected_token_id\tresolved_token\n")
        for text, tid in zip(selected_token_texts, selected_token_ids):
            resolved = tokenizer.convert_ids_to_tokens(tid)
            f.write(f"{text}\t{tid}\t{resolved}\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")

    emb = model.get_input_embeddings().weight.detach() if args.embedding_source == "input" else model.lm_head.weight.detach()
    if args.max_vocab > 0:
        emb = emb[: args.max_vocab]

    emb = emb.to(device=args.device, dtype=_dtype(args.dtype))
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    requested_tokens = [x.strip() for x in args.tokens.split(",") if x.strip()]
    selected_token_ids = _resolve_token_ids(tokenizer, requested_tokens)
    for token_id in selected_token_ids:
        if token_id >= emb.size(0):
            raise ValueError(f"Selected token id {token_id} is outside analyzed vocab size {emb.size(0)}")

    angles = _write_selected_token_csvs(
        output_dir=output_dir,
        tokenizer=tokenizer,
        emb=emb,
        selected_token_ids=selected_token_ids,
        selected_token_texts=requested_tokens,
        top_k=args.top_k,
    )
    _write_plotly_dashboard(output_dir, tokenizer, selected_token_ids, requested_tokens, angles, args.max_plot_vocab)

    threshold_cos = math.cos(math.radians(args.angle_threshold_deg))
    component_ids, degrees = _build_islands(emb, threshold_cos, args.chunk_size)
    component_sizes = np.bincount(component_ids)
    _write_island_files(output_dir, tokenizer, component_ids, component_sizes, degrees, args.min_island_size)

    with (output_dir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"embedding_source={args.embedding_source}\n")
        f.write(f"vocab_size={emb.size(0)}\n")
        f.write(f"angle_threshold_deg={args.angle_threshold_deg}\n")
        f.write(f"num_components={int(component_sizes.shape[0])}\n")
        f.write(f"largest_component={int(component_sizes.max())}\n")
        f.write(f"selected_tokens={','.join(requested_tokens)}\n")

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

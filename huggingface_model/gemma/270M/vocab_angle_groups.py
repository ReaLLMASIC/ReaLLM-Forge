#!/usr/bin/env python3
"""Analyze Gemma vocabulary embedding angle graph and connected groups.

Builds an undirected graph over vocab tokens where an edge exists when the
pairwise angle is <= threshold_degrees. Produces CSV summaries, plots, and a
focused report for digit tokens 0-9.
"""
from __future__ import annotations

import argparse
import math
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma vocab angle component analysis")
    parser.add_argument("--model", default="google/gemma-3-270m", help="HF model name")
    parser.add_argument(
        "--embedding-source",
        choices=["input", "lm_head"],
        default="input",
        help="Use model input embeddings or lm_head weights",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=70.0,
        help="Create graph edge when pairwise angle <= this threshold",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=-1,
        help="Limit vocab size for analysis (-1 for full vocab)",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Block size for similarity scan")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device for similarity scan (default cpu to stay RAM-safe)",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", default="./gemma_vocab_angle_groups")
    parser.add_argument(
        "--write-token-assignments",
        action="store_true",
        help="Write full token->component CSV (can be large)",
    )
    return parser.parse_args()


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


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _digit_token_map(tokenizer: AutoTokenizer, vocab_size: int) -> tuple[dict[str, list[int]], dict[int, str]]:
    digit_map: dict[str, list[int]] = {str(d): [] for d in range(10)}
    token_text: dict[int, str] = {}
    for idx in range(vocab_size):
        tok = tokenizer.convert_ids_to_tokens(idx)
        cleaned = tok.replace("▁", "").strip()
        if len(cleaned) != 1:
            continue
        if cleaned in digit_map:
            digit_key = cleaned
        else:
            try:
                numeric_value = unicodedata.digit(cleaned)
                digit_key = str(int(numeric_value))
            except (TypeError, ValueError):
                continue
            if digit_key not in digit_map:
                continue
        digit_map[digit_key].append(idx)
        token_text[idx] = tok
    return digit_map, token_text


def _render_progress(
    processed: int,
    total: int,
    start_time: float,
    edges: int,
) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = processed / elapsed
    remaining = total - processed
    eta_sec = remaining / max(rate, 1e-9)
    pct = (processed / max(total, 1)) * 100.0
    print(
        f"\rBlocks: {processed}/{total} ({pct:5.1f}%) | "
        f"Edges: {edges:,} | ETA: {eta_sec:7.1f}s",
        end="",
        flush=True,
    )


def _compute_edges(
    emb: torch.Tensor,
    cosine_threshold: float,
    chunk_size: int,
) -> tuple[UnionFind, np.ndarray, int]:
    n = emb.size(0)
    degrees = np.zeros(n, dtype=np.int64)
    uf = UnionFind(n)
    edge_count = 0
    num_blocks = math.ceil(n / chunk_size)
    total_block_pairs = num_blocks * (num_blocks + 1) // 2
    processed_pairs = 0
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
            if pairs.numel() == 0:
                processed_pairs += 1
                if processed_pairs % 10 == 0 or processed_pairs == total_block_pairs:
                    _render_progress(processed_pairs, total_block_pairs, started, edge_count)
                continue

            src = (pairs[:, 0] + i0).cpu().numpy().astype(np.int32)
            dst = (pairs[:, 1] + j0).cpu().numpy().astype(np.int32)
            edge_count += int(src.shape[0])
            degrees += np.bincount(src, minlength=n)
            degrees += np.bincount(dst, minlength=n)
            for s, d in zip(src, dst):
                uf.union(int(s), int(d))

            processed_pairs += 1
            if processed_pairs % 10 == 0 or processed_pairs == total_block_pairs:
                _render_progress(processed_pairs, total_block_pairs, started, edge_count)

    print()
    return uf, degrees, edge_count


def _write_component_summary(
    output_dir: Path,
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
) -> None:
    counts = np.bincount(component_ids, minlength=component_sizes.shape[0])
    ordered = np.argsort(counts)[::-1]
    with (output_dir / "component_summary.csv").open("w", encoding="utf-8") as f:
        f.write("component_id,size\n")
        for cid in ordered:
            f.write(f"{cid},{int(counts[cid])}\n")


def _write_digit_reports(
    output_dir: Path,
    digit_map: dict[str, list[int]],
    digit_token_text: dict[int, str],
    component_ids: np.ndarray,
    component_sizes: np.ndarray,
    degrees: np.ndarray,
) -> None:
    rows: list[tuple[str, int, str, int, int, str]] = []
    for d in range(10):
        digit = str(d)
        ids = digit_map[digit]
        if not ids:
            rows.append((digit, 0, "", -1, 0, ""))
            continue
        for token_id in ids:
            comp = int(component_ids[token_id])
            rows.append(
                (
                    digit,
                    len(ids),
                    digit_token_text.get(token_id, ""),
                    comp,
                    int(component_sizes[comp]),
                    str(int(degrees[token_id])),
                )
            )

    with (output_dir / "digit_report.csv").open("w", encoding="utf-8") as f:
        f.write("digit,num_digit_tokens,token,component_id,component_size,degree\n")
        for row in rows:
            f.write(",".join(map(str, row)).replace("\n", " ") + "\n")

    with (output_dir / "digit_report.md").open("w", encoding="utf-8") as f:
        f.write("# Digit Token Report (0-9)\n\n")
        f.write("| Digit | Num Digit Tokens | Token | Component ID | Component Size | Degree |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row[0]} | {row[1]} | `{row[2]}` | {row[3]} | {row[4]} | {row[5]} |\n"
            )


def _make_plots(output_dir: Path, component_sizes: np.ndarray, degrees: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    sorted_sizes = np.sort(component_sizes)[::-1]

    plt.figure(figsize=(10, 5))
    top_n = min(30, len(sorted_sizes))
    plt.bar(np.arange(top_n), sorted_sizes[:top_n])
    plt.title("Top Connected Component Sizes")
    plt.xlabel("Component Rank")
    plt.ylabel("Size")
    plt.tight_layout()
    plt.savefig(output_dir / "component_sizes_top30.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(component_sizes, bins=50, log=True)
    plt.title("Distribution of Connected Component Sizes")
    plt.xlabel("Component Size")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "component_size_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(degrees, bins=60, log=True)
    plt.title("Degree Distribution")
    plt.xlabel("Node Degree")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "degree_hist.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = _torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")

    if args.embedding_source == "input":
        emb = model.get_input_embeddings().weight.detach()
    else:
        emb = model.lm_head.weight.detach()

    if args.max_vocab > 0:
        emb = emb[: args.max_vocab]

    emb = emb.to(device=args.device, dtype=dtype)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cosine_threshold = math.cos(math.radians(args.angle_threshold_deg))
    uf, degrees, edge_count = _compute_edges(emb, cosine_threshold=cosine_threshold, chunk_size=args.chunk_size)

    n = emb.size(0)
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
    _, component_ids = np.unique(roots, return_inverse=True)
    component_ids = component_ids.astype(np.int32)
    component_sizes = np.bincount(component_ids).astype(np.int32)
    digit_map, digit_token_text = _digit_token_map(tokenizer, n)

    with (output_dir / "analysis_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"embedding_source={args.embedding_source}\n")
        f.write(f"vocab_size_analyzed={n}\n")
        f.write(f"angle_threshold_deg={args.angle_threshold_deg}\n")
        f.write(f"cosine_threshold={cosine_threshold:.8f}\n")
        f.write(f"num_edges={int(edge_count)}\n")
        f.write(f"num_components={int(component_sizes.shape[0])}\n")
        f.write(f"largest_component={int(component_sizes.max())}\n")
        f.write(f"isolated_tokens={(degrees == 0).sum()}\n")

    _write_component_summary(output_dir, component_ids, component_sizes)
    _write_digit_reports(output_dir, digit_map, digit_token_text, component_ids, component_sizes, degrees)

    if args.write_token_assignments:
        with (output_dir / "token_component_assignments.csv").open("w", encoding="utf-8") as f:
            f.write("token_id,token,component_id,component_size,degree\n")
            for token_id in range(n):
                token = tokenizer.convert_ids_to_tokens(token_id).replace("\n", "\\n")
                cid = int(component_ids[token_id])
                f.write(f"{token_id},{token},{cid},{int(component_sizes[cid])},{int(degrees[token_id])}\n")

    _make_plots(output_dir, component_sizes, degrees)
    print(f"Done. Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

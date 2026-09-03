import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build island-routing augmentation metadata from island analysis and "
            "benchmark dense vs routed matvec speed (TTFT/decode proxy)."
        )
    )
    p.add_argument("ckpt_dir", type=str, help="Directory containing ckpt.pt")
    p.add_argument(
        "--island_json",
        type=str,
        default=None,
        help="Path to islands_detailed.json (default: <ckpt_dir>/island_analysis/islands_detailed.json)",
    )
    p.add_argument("--threshold", type=float, required=True, help="Threshold to select from threshold_tries")
    p.add_argument(
        "--provider_mode",
        choices=["top", "mean"],
        default="top",
        help="How to choose representative vectors used in replacement metadata",
    )
    p.add_argument(
        "--probe_mode",
        choices=["mean", "top"],
        default="mean",
        help="How to build probe vectors for routing decisions (default mean of each island)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <ckpt_dir>/island_routing)",
    )
    p.add_argument("--bench_repeats", type=int, default=200, help="Benchmark repeats per tensor")
    p.add_argument("--warmup", type=int, default=30, help="Warmup iterations per tensor")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_threshold_entry(report: Dict, target_threshold: float) -> Optional[Dict]:
    best = None
    best_dist = 1e9
    for entry in report.get("threshold_tries", []):
        dist = abs(float(entry["threshold"]) - target_threshold)
        if dist < best_dist:
            best_dist = dist
            best = entry
    return best


def build_rep_vector(weight: torch.Tensor, island: Dict, mode: str) -> torch.Tensor:
    members = island["member_indices"]
    if mode == "mean":
        return weight[members].mean(dim=0)
    providers = island.get("providers", [])
    if providers:
        return weight[int(providers[0]["vector_index"])]
    return weight[members].mean(dim=0)


def routed_matvec(
    x: torch.Tensor,
    weight: torch.Tensor,
    probe_vectors: torch.Tensor,
    island_indices: List[List[int]],
    fallback_indices: List[int],
) -> torch.Tensor:
    # Stage 1: probe vectors (islands + fallback uncovered group)
    scores = x @ probe_vectors.T
    chosen = int(torch.argmax(scores).item())

    if chosen < len(island_indices):
        idx_list = island_indices[chosen]
    else:
        idx_list = fallback_indices

    idx = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
    out = torch.zeros(weight.shape[0], dtype=x.dtype, device=weight.device)
    if idx.numel() > 0:
        out[idx] = weight[idx] @ x
    return out


def benchmark_tensor(
    weight: torch.Tensor,
    probe_vectors: torch.Tensor,
    island_indices: List[List[int]],
    fallback_indices: List[int],
    warmup: int,
    repeats: int,
    gen: torch.Generator,
) -> Dict:
    in_dim = weight.shape[1]
    x = torch.randn(in_dim, generator=gen, dtype=weight.dtype)

    for _ in range(warmup):
        _ = weight @ x
        _ = routed_matvec(x, weight, probe_vectors, island_indices, fallback_indices)

    t0 = time.perf_counter(); _ = weight @ x; t1 = time.perf_counter()
    baseline_ttft = t1 - t0

    t0 = time.perf_counter(); _ = routed_matvec(x, weight, probe_vectors, island_indices, fallback_indices); t1 = time.perf_counter()
    routed_ttft = t1 - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = weight @ x
    t1 = time.perf_counter()
    baseline_decode_s = t1 - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = routed_matvec(x, weight, probe_vectors, island_indices, fallback_indices)
    t1 = time.perf_counter()
    routed_decode_s = t1 - t0

    return {
        "ttft_baseline_ms": baseline_ttft * 1e3,
        "ttft_routed_ms": routed_ttft * 1e3,
        "decode_baseline_tok_s": repeats / max(baseline_decode_s, 1e-9),
        "decode_routed_tok_s": repeats / max(routed_decode_s, 1e-9),
    }


def write_plotly_speed_html(rows: List[Dict], out_html: str):
    import plotly.graph_objects as go

    tensors = [r["tensor"] for r in rows]
    ttft_b = [r["ttft_baseline_ms"] for r in rows]
    ttft_r = [r["ttft_routed_ms"] for r in rows]
    dec_b = [r["decode_baseline_tok_s"] for r in rows]
    dec_r = [r["decode_routed_tok_s"] for r in rows]

    fig = go.Figure()
    fig.add_bar(name="TTFT baseline (ms)", x=tensors, y=ttft_b)
    fig.add_bar(name="TTFT routed (ms)", x=tensors, y=ttft_r)
    fig.add_bar(name="Decode baseline (tok/s)", x=tensors, y=dec_b, yaxis="y2")
    fig.add_bar(name="Decode routed (tok/s)", x=tensors, y=dec_r, yaxis="y2")
    fig.update_layout(
        barmode="group",
        title="Island routing speed comparison (per tensor)",
        xaxis_title="Tensor",
        yaxis=dict(title="TTFT (ms)"),
        yaxis2=dict(title="Decode speed (tok/s)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    island_json = args.island_json or os.path.join(args.ckpt_dir, "island_analysis", "islands_detailed.json")
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "island_routing")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

    with open(island_json, "r", encoding="utf-8") as f:
        island_payload = json.load(f)

    report_by_tensor = {r["tensor"]: r for r in island_payload.get("reports", [])}

    routing_payload = {
        "ckpt_dir": args.ckpt_dir,
        "source_island_json": island_json,
        "threshold": args.threshold,
        "provider_mode": args.provider_mode,
        "probe_mode": args.probe_mode,
        "tensors": {},
    }

    gen = torch.Generator().manual_seed(args.seed)
    speed_rows = []

    for tensor_name, report in report_by_tensor.items():
        if tensor_name not in state_dict:
            continue
        w = state_dict[tensor_name]
        if not torch.is_floating_point(w) or w.ndim != 2:
            continue

        entry = get_threshold_entry(report, args.threshold)
        if entry is None:
            continue
        islands = entry.get("islands", [])
        if not islands:
            continue

        island_member_lists = [x["member_indices"] for x in islands]

        replacement_reps = torch.stack([build_rep_vector(w, island, args.provider_mode) for island in islands], dim=0)
        island_probe_vectors = torch.stack([build_rep_vector(w, island, args.probe_mode) for island in islands], dim=0)

        covered = set()
        for members in island_member_lists:
            covered.update(int(i) for i in members)
        all_rows = set(range(w.shape[0]))
        uncovered = sorted(all_rows - covered)

        if uncovered:
            fallback_probe = w[uncovered].mean(dim=0)
            fallback_indices = uncovered
        else:
            # fallback still exists to guarantee a valid route target
            fallback_probe = w.mean(dim=0)
            fallback_indices = list(range(w.shape[0]))

        probe_vectors = torch.cat([island_probe_vectors, fallback_probe.unsqueeze(0)], dim=0)

        routing_payload["tensors"][tensor_name] = {
            "shape": list(w.shape),
            "num_islands": len(island_member_lists),
            "island_sizes": [len(x) for x in island_member_lists],
            "replacement_representatives": replacement_reps,
            "probe_vectors": probe_vectors,
            "island_indices": island_member_lists,
            "fallback_indices": fallback_indices,
            "covered_rows": len(covered),
            "uncovered_rows": len(uncovered),
        }

        bench = benchmark_tensor(
            w.float(),
            probe_vectors.float(),
            island_member_lists,
            fallback_indices,
            args.warmup,
            args.bench_repeats,
            gen,
        )
        speed_rows.append(
            {
                "tensor": tensor_name,
                "num_islands": len(island_member_lists),
                "uncovered_rows": len(uncovered),
                **bench,
                "ttft_speedup": bench["ttft_baseline_ms"] / max(bench["ttft_routed_ms"], 1e-9),
                "decode_speedup": bench["decode_routed_tok_s"] / max(bench["decode_baseline_tok_s"], 1e-9),
            }
        )

    routing_pt = os.path.join(out_dir, "island_routing.pt")
    torch.save(routing_payload, routing_pt)

    csv_path = os.path.join(out_dir, "island_routing_speed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "tensor", "num_islands", "uncovered_rows",
            "ttft_baseline_ms", "ttft_routed_ms", "ttft_speedup",
            "decode_baseline_tok_s", "decode_routed_tok_s", "decode_speedup",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in speed_rows:
            writer.writerow(row)

    json_path = os.path.join(out_dir, "island_routing_speed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": speed_rows}, f, indent=2)

    html_path = os.path.join(out_dir, "island_routing_speed.html")
    if speed_rows:
        try:
            write_plotly_speed_html(speed_rows, html_path)
            html_status = f"Wrote {html_path}"
        except Exception as exc:
            html_status = f"Skipped HTML generation ({exc})"
    else:
        html_status = "Skipped HTML generation (no eligible tensors/islands)"

    print(f"Wrote {routing_pt}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(html_status)


if __name__ == "__main__":
    main()

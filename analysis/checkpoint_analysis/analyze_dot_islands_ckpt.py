import argparse
import csv
import json
import os
import re
from collections import deque
from typing import Dict, List

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze checkpoint tensors and find vector 'islands' using pairwise "
            "dot/cosine similarity across threshold tries."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt from a training run",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional regex to filter tensor names",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity metric used to build pairwise matrix",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.2,0.35,0.5",
        help="Comma-separated thresholds to try when finding islands",
    )
    parser.add_argument(
        "--min_island_size",
        type=int,
        default=3,
        help="Minimum connected-component size to report",
    )
    parser.add_argument(
        "--top_providers",
        type=int,
        default=5,
        help="Number of top provider vectors to print per island",
    )
    parser.add_argument(
        "--max_vectors",
        type=int,
        default=2048,
        help="Max vectors per tensor; larger tensors are randomly subsampled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for JSON/CSV/HTML artifacts",
    )
    return parser.parse_args()


def tensor_to_vectors(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return torch.empty(0, 0)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def connected_components(mask: torch.Tensor) -> List[List[int]]:
    n = mask.shape[0]
    visited = torch.zeros(n, dtype=torch.bool)
    islands: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        comp: List[int] = []

        while queue:
            node = queue.popleft()
            comp.append(node)
            neighbors = torch.nonzero(mask[node], as_tuple=False).view(-1).tolist()
            for nxt in neighbors:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)

        islands.append(comp)

    return islands


def compute_similarity(vectors: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors @ vectors.T


def summarize_islands(islands: List[List[int]], num_vectors: int) -> Dict[str, float]:
    if not islands:
        return {
            "num_islands": 0,
            "largest_island_size": 0,
            "mean_island_size": 0.0,
            "coverage_ratio": 0.0,
        }

    sizes = [len(x) for x in islands]
    covered = sum(sizes)
    return {
        "num_islands": len(islands),
        "largest_island_size": max(sizes),
        "mean_island_size": float(sum(sizes) / len(sizes)),
        "coverage_ratio": float(covered / max(num_vectors, 1)),
    }


def analyze_tensor(
    name: str,
    vectors: torch.Tensor,
    thresholds: List[float],
    metric: str,
    min_island_size: int,
    top_providers: int,
):
    sim = compute_similarity(vectors, metric=metric)
    sim.fill_diagonal_(0)

    tensor_report = {
        "tensor": name,
        "num_vectors": vectors.shape[0],
        "vector_dim": vectors.shape[1],
        "threshold_tries": [],
    }

    for threshold in thresholds:
        mask = sim >= threshold
        islands = connected_components(mask)
        islands = [comp for comp in islands if len(comp) >= min_island_size]

        island_reports = []
        for island in islands:
            island_tensor = sim[island][:, island]
            provider_scores = island_tensor.mean(dim=1)
            best_scores, best_local_idx = torch.topk(
                provider_scores,
                k=min(top_providers, len(island)),
                largest=True,
            )

            providers = []
            for score, local_idx in zip(best_scores.tolist(), best_local_idx.tolist()):
                global_idx = island[local_idx]
                providers.append(
                    {
                        "vector_index": int(global_idx),
                        "mean_similarity_to_island": float(score),
                    }
                )

            island_reports.append(
                {
                    "size": len(island),
                    "member_indices": [int(i) for i in island],
                    "providers": providers,
                }
            )

        summary = summarize_islands(island_reports_to_members(island_reports), vectors.shape[0])
        tensor_report["threshold_tries"].append(
            {
                "threshold": threshold,
                "summary": summary,
                "islands": island_reports,
            }
        )

    return tensor_report


def island_reports_to_members(island_reports: List[Dict]) -> List[List[int]]:
    return [island["member_indices"] for island in island_reports]


def write_csv_summary(reports: List[Dict], csv_path: str):
    fields = [
        "tensor",
        "num_vectors",
        "vector_dim",
        "threshold",
        "num_islands",
        "largest_island_size",
        "mean_island_size",
        "coverage_ratio",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for ttry in report["threshold_tries"]:
                row = {
                    "tensor": report["tensor"],
                    "num_vectors": report["num_vectors"],
                    "vector_dim": report["vector_dim"],
                    "threshold": ttry["threshold"],
                    **ttry["summary"],
                }
                writer.writerow(row)


def write_plotly_html(reports: List[Dict], html_path: str):
    import plotly.graph_objects as go

    fig = go.Figure()
    tensor_names = [r["tensor"] for r in reports]

    visible_mask = []
    for i, report in enumerate(reports):
        x_vals = [x["threshold"] for x in report["threshold_tries"]]
        num_islands = [x["summary"]["num_islands"] for x in report["threshold_tries"]]
        largest = [x["summary"]["largest_island_size"] for x in report["threshold_tries"]]
        coverage = [x["summary"]["coverage_ratio"] for x in report["threshold_tries"]]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=num_islands,
                mode="lines+markers",
                name=f"{report['tensor']} | num_islands",
                yaxis="y1",
                visible=(i == 0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=largest,
                mode="lines+markers",
                name=f"{report['tensor']} | largest_island",
                yaxis="y1",
                visible=(i == 0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=coverage,
                mode="lines+markers",
                name=f"{report['tensor']} | coverage",
                yaxis="y2",
                visible=(i == 0),
            )
        )
        visible_mask.append([False] * (3 * len(reports)))

    for i in range(len(reports)):
        visible_mask[i][3 * i] = True
        visible_mask[i][3 * i + 1] = True
        visible_mask[i][3 * i + 2] = True

    buttons = []
    for i, name in enumerate(tensor_names):
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": visible_mask[i]},
                    {"title": f"Island summary vs threshold: {name}"},
                ],
            }
        )

    fig.update_layout(
        title=f"Island summary vs threshold: {tensor_names[0] if tensor_names else 'N/A'}",
        xaxis_title="Threshold",
        yaxis=dict(title="Island Count / Largest Size"),
        yaxis2=dict(title="Coverage Ratio", overlaying="y", side="right", range=[0, 1]),
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.02,
                "xanchor": "left",
                "y": 1,
                "yanchor": "top",
            }
        ],
        legend=dict(orientation="h"),
    )

    fig.write_html(html_path, include_plotlyjs="cdn")


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        raise ValueError("No valid thresholds provided")

    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "island_analysis")
    os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(args.pattern) if args.pattern else None

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

    generator = torch.Generator().manual_seed(args.seed)

    reports = []
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            continue
        if pattern and pattern.search(key) is None:
            continue

        vectors = tensor_to_vectors(tensor.detach().float())
        if vectors.numel() == 0 or vectors.shape[0] < 2:
            continue

        sampled_indices = None
        original_num_vectors = vectors.shape[0]
        if vectors.shape[0] > args.max_vectors:
            sampled_indices = torch.randperm(vectors.shape[0], generator=generator)[: args.max_vectors]
            vectors = vectors[sampled_indices]

        report = analyze_tensor(
            name=key,
            vectors=vectors,
            thresholds=thresholds,
            metric=args.metric,
            min_island_size=args.min_island_size,
            top_providers=args.top_providers,
        )

        if sampled_indices is not None:
            report["sampled_from"] = int(original_num_vectors)
            report["sampled_indices"] = [int(i) for i in sampled_indices.tolist()]

        reports.append(report)

    payload = {
        "ckpt_dir": args.ckpt_dir,
        "metric": args.metric,
        "thresholds": thresholds,
        "min_island_size": args.min_island_size,
        "top_providers": args.top_providers,
        "num_tensors_analyzed": len(reports),
        "reports": reports,
    }

    json_path = os.path.join(out_dir, "islands_detailed.json")
    csv_path = os.path.join(out_dir, "islands_summary.csv")
    html_path = os.path.join(out_dir, "islands_dashboard.html")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv_summary(reports, csv_path)

    if reports:
        try:
            write_plotly_html(reports, html_path)
            html_status = f"Wrote {html_path}"
        except Exception as exc:
            html_status = f"Skipped HTML generation ({exc})"
    else:
        html_status = "Skipped HTML generation (no tensors matched filters)"

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(html_status)


if __name__ == "__main__":
    main()

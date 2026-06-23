# utils/min_angle_graph_export.py
import csv
import json
import os

import torch
import torch.nn.functional as F


def resolve_min_angle_graph_device(weight, requested_device="auto"):
    """Choose the compute device for blockwise minimum-angle graph export."""
    if requested_device == "auto":
        if weight.is_cuda:
            return weight.device
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested for minimum-angle graph export but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def compute_min_angle_graph(weight, block_size=2048, compute_device="auto"):
    """Compute each row vector's closest non-self row by signed angular distance."""
    compute_device = resolve_min_angle_graph_device(weight, compute_device)
    block_size = max(1, int(block_size))
    vocab_size = weight.shape[0]

    with torch.no_grad():
        weight = weight.detach()
        norms = torch.linalg.vector_norm(weight, ord=2, dim=1).detach().cpu()
        best_cosine = torch.full((vocab_size,), -float("inf"), device=compute_device)
        best_other_token_id = torch.full((vocab_size,), -1, dtype=torch.long, device=compute_device)

        for row_start in range(0, vocab_size, block_size):
            row_end = min(row_start + block_size, vocab_size)
            row_block = weight[row_start:row_end].to(compute_device, non_blocking=True)
            row_block = F.normalize(row_block, p=2, dim=1)

            for col_start in range(0, vocab_size, block_size):
                col_end = min(col_start + block_size, vocab_size)
                col_block = weight[col_start:col_end].to(compute_device, non_blocking=True)
                col_block = F.normalize(col_block, p=2, dim=1)
                cosine_block = row_block @ col_block.T

                if row_start < col_end and col_start < row_end:
                    diag_start = max(row_start, col_start)
                    diag_end = min(row_end, col_end)
                    diag_rows = torch.arange(
                        diag_start - row_start,
                        diag_end - row_start,
                        device=compute_device,
                    )
                    diag_cols = torch.arange(
                        diag_start - col_start,
                        diag_end - col_start,
                        device=compute_device,
                    )
                    cosine_block[diag_rows, diag_cols] = -float("inf")

                block_best_cosine, block_best_col = cosine_block.max(dim=1)
                current_best = best_cosine[row_start:row_end]
                current_other = best_other_token_id[row_start:row_end]
                update_mask = block_best_cosine > current_best
                current_best[update_mask] = block_best_cosine[update_mask]
                current_other[update_mask] = block_best_col[update_mask] + col_start

        best_cosine = best_cosine.clamp(-1.0, 1.0).cpu()
        best_other_token_id = best_other_token_id.cpu()
        min_angles = torch.rad2deg(torch.acos(best_cosine))
        sorted_rank = torch.argsort(min_angles)
        rank_by_token = torch.empty_like(sorted_rank)
        rank_by_token[sorted_rank] = torch.arange(vocab_size, dtype=torch.long)

    return {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "compute_device": str(compute_device),
        "norms": norms,
        "best_cosine": best_cosine,
        "best_other_token_id": best_other_token_id,
        "min_angles": min_angles,
        "rank_by_token": rank_by_token,
    }


def safe_filename_label(label):
    """Return a filesystem-friendly label while preserving readable separators."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def write_min_angle_graph_export(graph, export_dir, label, iter_num, val_loss):
    """Write a minimum-angle graph CSV plus JSON sidecar and return both paths."""
    export_dir = os.path.expanduser(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    safe_label = safe_filename_label(label)
    file_stem = f"{safe_label}_iter_{iter_num:08d}_val_{val_loss:.6f}"
    csv_path = os.path.join(export_dir, f"{file_stem}.csv")
    json_path = os.path.join(export_dir, f"{file_stem}.json")

    norms = graph["norms"]
    best_cosine = graph["best_cosine"]
    best_other_token_id = graph["best_other_token_id"]
    min_angles = graph["min_angles"]
    rank_by_token = graph["rank_by_token"]
    vocab_size = graph["vocab_size"]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "token_id",
            "nearest_token_id",
            "min_angle_deg",
            "cosine",
            "token_vector_length",
            "nearest_token_vector_length",
            "min_angle_rank",
        ])
        for token_id in range(vocab_size):
            other_id = int(best_other_token_id[token_id])
            writer.writerow([
                token_id,
                other_id,
                f"{float(min_angles[token_id]):.9f}",
                f"{float(best_cosine[token_id]):.9f}",
                f"{float(norms[token_id]):.9f}",
                f"{float(norms[other_id]):.9f}" if other_id >= 0 else "",
                int(rank_by_token[token_id]),
            ])

    metadata = {
        "iter_num": iter_num,
        "val_loss": val_loss,
        "label": label,
        "csv_path": csv_path,
        "vocab_size": vocab_size,
        "block_size": graph["block_size"],
        "compute_device": graph["compute_device"],
        "angle_definition": "signed 0-180 degrees, closest non-self token by maximum cosine",
    }
    with open(json_path, "w") as json_file:
        json.dump(metadata, json_file, indent=2)

    return csv_path, json_path


def export_min_angle_graph(weight, export_dir, label, iter_num, val_loss, block_size=2048, compute_device="auto"):
    """Compute and write the LM-head minimum-angle graph export."""
    graph = compute_min_angle_graph(weight, block_size=block_size, compute_device=compute_device)
    return write_min_angle_graph_export(graph, export_dir, label, iter_num, val_loss)

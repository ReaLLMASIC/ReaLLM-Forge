#!/usr/bin/env python3
"""Export token trajectories, projecting higher-dimensional embeddings to 3D."""

import argparse
import json
import math
import pickle
import re
from pathlib import Path

import torch


def iteration(path: Path) -> int:
    if path.name == "ckpt.pt":
        return 10**18
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def project_to_3d(frame_vectors: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
    """Use one global PCA basis so positions remain comparable across time."""
    embedding_dim = frame_vectors[0].shape[1]
    if embedding_dim == 2:
        projected = [torch.nn.functional.pad(vectors, (0, 1)) for vectors in frame_vectors]
        return projected, {"method": "native_2d", "input_dimensions": 2}
    if embedding_dim == 3:
        return frame_vectors, {"method": "native", "input_dimensions": 3}
    if embedding_dim < 3:
        raise ValueError(f"expected at least 2 embedding dimensions, got {embedding_dim}")

    stacked = torch.cat(frame_vectors, dim=0).double()
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:3].T
    # SVD component signs are arbitrary. Fix them for reproducible JSON output.
    for column in range(components.shape[1]):
        pivot = components[:, column].abs().argmax()
        if components[pivot, column] < 0:
            components[:, column].neg_()
    projected = [(vectors.double() - mean).matmul(components).float() for vectors in frame_vectors]
    total_variance = singular_values.square().sum()
    explained = singular_values[:3].square() / total_variance.clamp_min(torch.finfo(torch.float64).eps)
    return projected, {
        "method": "pca",
        "input_dimensions": embedding_dim,
        "explained_variance_ratio": explained.tolist(),
        "fit": "all tokens across all checkpoint frames",
    }


def finite_metric(value):
    """Return a JSON-safe finite float or None for missing/legacy metrics."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def export(
    checkpoint_dir: Path,
    meta_path: Path,
    output: Path,
    dropout_iteration: int | None = None,
    transition_mode: str | None = None,
    transition_iterations: list[int] | None = None,
    affected_tokens: list[str] | None = None,
    duty_cycle: int | None = None,
) -> None:
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    tokens = [meta["itos"][i] for i in range(meta["vocab_size"])]

    candidates = sorted(checkpoint_dir.glob("*.pt"), key=iteration)
    if not candidates:
        raise FileNotFoundError(f"no .pt checkpoints found in {checkpoint_dir}")

    frame_iterations = []
    frame_vectors = []
    frame_metrics = []
    seen_iterations = set()
    fixed_norm = None
    wte_weight_tying = None
    optimizer_metadata = None
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_args = checkpoint.get("model_args", {})
        if optimizer_metadata is None:
            config = checkpoint.get("config", {})
            optimizer_metadata = {
                "name": config.get("optimizer"),
                "weight_decay": config.get("weight_decay"),
                "muon_include_all_weights": config.get("muon_include_all_weights", False),
            }
        if wte_weight_tying is None:
            wte_weight_tying = model_args.get("wte_weight_tying", True)
        if fixed_norm is None and model_args.get("wte_fixed_norm", False):
            fixed_norm = model_args.get("wte_fixed_norm_value")
            if fixed_norm is None:
                fixed_norm = (model_args.get("n_embd_wte") or model_args["n_embd"]) ** 0.5
        step = int(checkpoint.get("iter_num", iteration(path)))
        if step in seen_iterations:
            continue
        state = checkpoint["model"]
        key = next((k for k in state if k.removeprefix("_orig_mod.") == "transformer.wte.weight"), None)
        if key is None:
            raise KeyError(f"transformer.wte.weight not found in {path}")
        vectors = state[key].detach().float()
        if vectors.ndim != 2 or vectors.shape[0] != len(tokens):
            raise ValueError(f"{path}: expected {len(tokens)} token vectors, got {tuple(vectors.shape)}")
        if frame_vectors and vectors.shape[1] != frame_vectors[0].shape[1]:
            raise ValueError(f"{path}: embedding dimension changed between checkpoints")
        frame_iterations.append(step)
        frame_vectors.append(vectors)
        metrics = checkpoint.get("metrics") or {}
        frame_metrics.append({
            "train_loss": finite_metric(metrics.get("train_loss")),
            "val_loss": finite_metric(metrics.get("val_loss")),
        })
        seen_iterations.add(step)

    projected, projection = project_to_3d(frame_vectors)
    frames = [
        {"iteration": step, "positions": vectors.tolist(), "metrics": metrics}
        for step, vectors, metrics in zip(frame_iterations, projected, frame_metrics)
    ]

    payload = {
        "tokens": tokens,
        "trained_tokens": meta.get("initial_trained_tokens", meta.get("trained_tokens", list("0123456789"))),
        "unseen_tokens": meta.get("unseen_tokens", list("abcd")),
        "dropped_tokens": affected_tokens if affected_tokens is not None else meta.get("dropped_tokens", []),
        "dropout_iteration": dropout_iteration,
        "transition_mode": transition_mode,
        "transition_iterations": transition_iterations or ([dropout_iteration] if dropout_iteration is not None else []),
        "duty_cycle": duty_cycle,
        "fixed_norm": fixed_norm,
        "wte_weight_tying": wte_weight_tying,
        "optimizer": optimizer_metadata,
        "projection": projection,
        "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(frames)} frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/digits_3d/meta.pkl"))
    parser.add_argument("--output", type=Path, default=Path("report/threejs/digits-3d/token_trajectories.json"))
    parser.add_argument("--dropout-iteration", type=int, default=None)
    parser.add_argument("--transition-mode", choices=["drop", "add", "duty_cycle"], default=None)
    parser.add_argument("--transition-iterations", type=int, nargs="*", default=None)
    parser.add_argument("--affected-tokens", default=None, help="Characters whose availability changes.")
    parser.add_argument("--duty-cycle", type=int, default=None)
    args = parser.parse_args()
    export(
        args.checkpoint_dir, args.meta, args.output, args.dropout_iteration,
        args.transition_mode, args.transition_iterations,
        list(args.affected_tokens) if args.affected_tokens is not None else None,
        args.duty_cycle,
    )


if __name__ == "__main__":
    main()

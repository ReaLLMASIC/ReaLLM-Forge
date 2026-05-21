#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer

PRESETS = {
    "digits": [str(i) for i in range(10)],
    "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "months": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantization angle analysis for token sets")
    p.add_argument("--model", default="google/gemma-3-270m")
    p.add_argument("--embedding-source", choices=["input", "lm_head"], default="input")
    p.add_argument("--device", default="cpu")
    p.add_argument("--preset", choices=["digits", "weekdays", "months"], default="digits")
    p.add_argument("--output-dir", default="./gemma_token_quant_angles")
    return p.parse_args()


def _resolve_ids(tokenizer: AutoTokenizer, tokens: list[str]) -> list[int]:
    out = []
    for t in tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid == tokenizer.unk_token_id:
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"Could not tokenize token {t}")
            tid = int(ids[0])
        out.append(int(tid))
    return out


def _symmetric_quantize(vecs: np.ndarray, mode: str) -> np.ndarray:
    x = vecs.copy()
    max_abs = np.max(np.abs(x), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    if mode == "binary":
        return np.where(x >= 0, 1.0, -1.0) * max_abs
    if mode == "ternary":
        thr = 0.5 * max_abs
        q = np.zeros_like(x)
        q[x > thr] = 1.0
        q[x < -thr] = -1.0
        return q * max_abs
    bits = int(mode)
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax)
    return q * scale


def _angles(vecs: np.ndarray) -> np.ndarray:
    n = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip(n @ n.T, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _write_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("token," + ",".join(labels) + "\n")
        for i, tok in enumerate(labels):
            f.write(tok + "," + ",".join(f"{matrix[i, j]:.6f}" for j in range(len(labels))) + "\n")


def _plot_tsne(out: Path, labels: list[str], vecs_by_mode: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(11, 7))
    for mode, vecs in vecs_by_mode.items():
        tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=max(2, min(5, len(labels)-1)))
        pts = tsne.fit_transform(vecs)
        plt.scatter(pts[:, 0], pts[:, 1], s=20, label=mode, alpha=0.7)
    plt.title("t-SNE approximate structure across quantization modes")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "tsne_structure_all_modes.png", dpi=180)
    plt.close()


def _plot_distortion_per_token(out: Path, labels: list[str], baseline: np.ndarray, angles_by_mode: dict[str, np.ndarray]) -> None:
    x = np.arange(len(labels))
    for i, tok in enumerate(labels):
        plt.figure(figsize=(11, 5))
        for mode, mat in angles_by_mode.items():
            if mode == "fp32":
                continue
            delta = mat[i] - baseline[i]
            plt.plot(x, delta, marker="o", label=mode)
        plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("Angular distortion (deg, signed)")
        plt.xlabel("Other token")
        plt.title(f"Angular distortion direction/magnitude for {tok}")
        plt.grid(alpha=0.3)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"distortion_{tok}.png", dpi=180)
        plt.close()


def _make_plotly_html(out: Path, labels: list[str], angles_by_mode: dict[str, np.ndarray]) -> None:
    modes = list(angles_by_mode.keys())
    baseline = angles_by_mode["fp32"]

    fig = go.Figure()
    for i, tok in enumerate(labels):
        for mode in modes:
            y = angles_by_mode[mode][i]
            rank_base = np.argsort(baseline[i])
            rank_cur = np.argsort(y)
            disorder = (rank_cur != rank_base).sum()
            name = f"{tok} | {mode} | disorder={int(disorder)}"
            visible = i == 0
            fig.add_trace(go.Scatter(x=labels, y=y, mode="lines+markers", name=name, visible=visible))

    buttons = []
    traces_per_token = len(modes)
    for i, tok in enumerate(labels):
        vis = [False] * (len(labels) * traces_per_token)
        start = i * traces_per_token
        for j in range(traces_per_token):
            vis[start + j] = True
        buttons.append(dict(label=tok, method="update", args=[{"visible": vis}, {"title": f"Relative angles for {tok}"}]))

    fig.update_layout(
        title=f"Relative angles for {labels[0]}",
        xaxis_title="Other token",
        yaxis_title="Angle (degrees)",
        updatemenus=[dict(type="dropdown", buttons=buttons, x=1.02, y=1.0)],
        margin=dict(l=40, r=220, t=60, b=120),
    )
    fig.write_html(str(out / "relative_angles_selector.html"), include_plotlyjs="cdn")


def run_analysis(preset: str, model: str, embedding_source: str, device: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, attn_implementation="eager")
    emb = m.get_input_embeddings().weight.detach() if embedding_source == "input" else m.lm_head.weight.detach()
    emb = emb.to(device, dtype=torch.float32)

    labels = PRESETS[preset]
    ids = _resolve_ids(tokenizer, labels)
    vecs = emb[ids].cpu().numpy()

    modes = ["fp32", "8", "7", "6", "5", "4", "3", "ternary", "binary"]
    vecs_by_mode: dict[str, np.ndarray] = {}
    angles_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        q = vecs if mode == "fp32" else _symmetric_quantize(vecs, mode)
        vecs_by_mode[mode] = q
        ang = _angles(q)
        angles_by_mode[mode] = ang
        _write_csv(output_dir / f"angles_{mode}.csv", labels, ang)

    with (output_dir / "token_ids.csv").open("w", encoding="utf-8") as f:
        f.write("token,token_id,resolved_token\n")
        for tok, tid in zip(labels, ids):
            f.write(f"{tok},{tid},{tokenizer.convert_ids_to_tokens(tid)}\n")

    _plot_tsne(output_dir, labels, vecs_by_mode)
    _plot_distortion_per_token(output_dir, labels, angles_by_mode["fp32"], angles_by_mode)
    _make_plotly_html(output_dir, labels, angles_by_mode)


def main() -> None:
    a = parse_args()
    run_analysis(a.preset, a.model, a.embedding_source, a.device, Path(a.output_dir))


if __name__ == "__main__":
    main()

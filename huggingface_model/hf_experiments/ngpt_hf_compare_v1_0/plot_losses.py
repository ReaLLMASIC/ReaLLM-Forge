"""One chart per metric; never fabricate or interpolate absent measurements."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_runs(root, smooth=1):
    root = Path(root)
    runs = []
    for file in sorted(root.glob("*/metrics.csv")):
        with file.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            manifest = json.loads((file.parent/"run.json").read_text())
            runs.append((file.parent.name, rows, manifest))
    if not runs:
        raise FileNotFoundError(f"No */metrics.csv under {root}")
    # A graph must not silently mix unlike data or model dimensions.
    match = None
    for _, _, manifest in runs:
        spec = manifest["spec"]
        key = {k: spec[k] for k in ("data_sha256", "width", "layers", "heads", "context",
                                    "batch_size", "accumulation", "precision", "weight_storage",
                                    "steps", "core_only", "data_seed")}
        if match is None:
            match = key
        elif key != match:
            raise ValueError("Incompatible comparison runs; use separate output directories.")
    out = root/"plots"
    out.mkdir(exist_ok=True)
    synthetic = any(m["synthetic"] for _, _, m in runs)
    suffix = "\nSYNTHETIC CPU/PLUMBING TEST — not a language-model benchmark" if synthetic else ""
    charts = [
        ("train_loss", "step", "Training minibatch loss", "training_loss_vs_iterations"),
        ("val_loss", "step", "Validation loss", "validation_loss_vs_iterations"),
        ("train_probe_loss", "step", "Fixed training-set probe loss", "train_probe_loss_vs_iterations"),
        ("val_loss", "elapsed_seconds", "Validation loss vs elapsed run time", "validation_loss_vs_seconds"),
    ]
    for metric, xkey, title, name in charts:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, rows, _ in runs:
            valid = [r for r in rows if r.get(metric) not in (None, "")]
            if not valid:
                continue
            x = np.array([float(r[xkey]) for r in valid])
            y = np.array([float(r[metric]) for r in valid])
            if metric == "train_loss" and smooth > 1:
                width = min(smooth, len(y))
                y = np.convolve(y, np.ones(width)/width, mode="valid")
                x = x[width-1:]
                label += f" (trailing mean {width})"
            ax.plot(x, y, label=label, linewidth=1.5)
        ax.set_title(title + suffix, fontsize=12)
        ax.set_xlabel("Completed optimizer updates" if xkey == "step" else "Elapsed seconds (includes evaluation/checkpoint overhead)")
        ax.set_ylabel("Next-token cross-entropy (nats/token)")
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out/f"{name}.png", dpi=160)
        fig.savefig(out/f"{name}.svg")
        plt.close(fig)
    print(f"Saved plots to {out}")

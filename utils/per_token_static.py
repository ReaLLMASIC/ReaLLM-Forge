"""Matplotlib static, vertically aligned per-token metric dashboards."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("training_seen_count", "Training occurrences"),
    ("val_loss", "Validation loss"),
    ("train_loss", "Sampled training loss"),
    ("vector_magnitude", "Token vector L2 magnitude"),
    ("min_pairwise_angle_deg", "Minimum pairwise angle (degrees)"),
)
ORDERINGS = (
    ("frequency", "training_seen_count"),
    ("validation_loss", "val_loss"),
    ("training_loss", "train_loss"),
    ("vector_magnitude", "vector_magnitude"),
    ("minimum_pairwise_angle", "min_pairwise_angle_deg"),
)


def _finite_range(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = float(values.min()), float(values.max())
    margin = (high - low) * 0.04 if high != low else max(abs(high) * 0.04, 0.04)
    return low - margin, high + margin


def write_static_dashboards(output_dir, rows):
    """Regenerate snapshot PNGs with dataset-wide, time-consistent y-axes."""
    paths = []
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        ranges = {metric: _finite_range(dataset_rows, metric) for metric, _ in METRICS}
        safe_dataset = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
        for iteration in sorted({row["iteration"] for row in dataset_rows}):
            snapshot = [row for row in dataset_rows if row["iteration"] == iteration]
            for order_label, sort_metric in ORDERINGS:
                ordered = sorted(
                    snapshot,
                    key=lambda row: (
                        np.isfinite(row[sort_metric]),
                        row[sort_metric] if np.isfinite(row[sort_metric]) else -np.inf,
                    ),
                    reverse=True,
                )
                fig, axes = plt.subplots(len(METRICS), 1, figsize=(18, 18), sharex=True)
                x = np.arange(len(ordered))
                for axis, (metric, metric_label) in zip(axes, METRICS):
                    axis.scatter(x, [row[metric] for row in ordered], s=7, alpha=0.8)
                    axis.set_ylabel(metric_label)
                    axis.grid(alpha=0.25)
                    if ranges[metric] is not None:
                        axis.set_ylim(*ranges[metric])
                axes[-1].set_xlabel(
                    f"Token rank, sorted high-to-low by {order_label.replace('_', ' ')}"
                )
                fig.suptitle(
                    f"{dataset}, iteration {iteration}: metrics sorted by "
                    f"{order_label.replace('_', ' ')}"
                )
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                filename = (
                    f"per_token_static_{safe_dataset}_iter_{iteration:08d}_"
                    f"by_{order_label}.png"
                )
                path = os.path.join(output_dir, filename)
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
    return paths

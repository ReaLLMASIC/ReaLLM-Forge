"""Compare fixed-budget test outcomes and matched-adaptation validation trajectories."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import anchor_dir, read_json, run_dir, write_json


def load_rows(path):
    if not Path(path).exists():
        return []
    # Only the last complete row per optimizer step is used after an interrupted write.
    out = {}
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["step"]] = row
    return [out[k] for k in sorted(out)]


def matched_point(rows, gain):
    """First temporal crossing; never sort a nonmonotone trajectory by gain."""
    if not rows:
        return None
    a0, b0 = rows[0]["a"]["loss"], rows[0]["b"]["loss"]
    points = [(b0 - r["b"]["loss"], r["a"]["loss"] - a0, r["step"]) for r in rows]
    for i, (g, f, s) in enumerate(points):
        if g >= gain:
            if i == 0:
                return {"forgetting": f, "step_estimate": s, "left_step": s, "right_step": s}
            g0, f0, s0 = points[i - 1]
            ratio = (gain - g0) / (g - g0)
            return {"forgetting": f0 + ratio * (f - f0), "step_estimate": s0 + ratio * (s - s0),
                    "left_step": s0, "right_step": s}
    return None


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def report(c, gains=(0.025, 0.05, 0.1, 0.2), plots=True):
    dest = Path(c["output"]) / "analysis"
    dest.mkdir(exist_ok=True)
    final, trajectories, matched = [], [], []
    available = {}
    for seed in c["seeds"]:
        anchor_test_path = anchor_dir(c, seed) / "test.json"
        if not anchor_test_path.exists():
            continue
        anchor_test = read_json(anchor_test_path)
        for lr in c["lrs"]:
            for mode in c["modes"]:
                root = run_dir(c, seed, mode, lr)
                rows = load_rows(root / "metrics.jsonl")
                if not rows:
                    continue
                available[(seed, lr, mode)] = rows
                a0, b0 = rows[0]["a"]["loss"], rows[0]["b"]["loss"]
                for r in rows:
                    entry = {"seed": seed, "lr": lr, "mode": mode, "step": r["step"],
                             "a_validation_loss": r["a"]["loss"], "b_validation_loss": r["b"]["loss"],
                             "forgetting": r["a"]["loss"] - a0, "gain": b0 - r["b"]["loss"],
                             "train_a_tokens": r["train_a_tokens"], "train_b_tokens": r["train_b_tokens"],
                             "train_seconds": r["train_seconds"], "peak_allocated_gib": r["peak_allocated_gib"]}
                    entry.update({f"swap_{k}": v for k, v in r.get("head_swap", {}).items()})
                    entry.update({f"head_{k}": v for k, v in r.get("head_geometry", {}).items()})
                    trajectories.append(entry)
                for gain in gains:
                    p = matched_point(rows, gain)
                    matched.append({"seed": seed, "lr": lr, "mode": mode, "gain_target": gain,
                                    "reached": p is not None, **(p or {})})
                if (root / "complete.json").exists():
                    test = read_json(root / "test.json")
                    info = read_json(root / "complete.json")
                    final.append({"seed": seed, "lr": lr, "mode": mode, "step": test["step"],
                                  "a_test_loss": test["a"]["loss"], "b_test_loss": test["b"]["loss"],
                                  "test_forgetting": test["a"]["loss"] - anchor_test["a"]["loss"],
                                  "test_gain": anchor_test["b"]["loss"] - test["b"]["loss"],
                                  "a_test_token_accuracy": test["a"]["token_accuracy"],
                                  "a_common_b_rare_test_loss": test["a"]["a_common_b_rare_loss"],
                                  "peak_allocated_gib": info["peak_allocated_gib"],
                                  "peak_reserved_gib": info["peak_reserved_gib"],
                                  "frozen_unchanged": info["frozen_unchanged"]})
    csv_write(dest / "trajectories.csv", trajectories)
    csv_write(dest / "final_test.csv", final)
    csv_write(dest / "matched_validation.csv", matched)
    paired = []
    for lr in c["lrs"]:
        for mode in c["modes"]:
            if mode == "full":
                continue
            for gain in gains:
                differences = []
                for seed in c["seeds"]:
                    base = matched_point(available.get((seed, lr, "full"), []), gain)
                    other = matched_point(available.get((seed, lr, mode), []), gain)
                    if base is not None and other is not None:
                        differences.append(base["forgetting"] - other["forgetting"])
                if differences:
                    d = np.asarray(differences)
                    ci = [None, None]
                    if len(d) >= 3:
                        samples = np.random.default_rng(2026).choice(d, (5000, len(d)), replace=True).mean(1)
                        ci = np.quantile(samples, [0.025, 0.975]).tolist()
                    paired.append({"mode": mode, "lr": lr, "gain_target": gain, "paired_seeds": len(d),
                                   "mean_forgetting_reduction": d.mean().item(),
                                   "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1]})
    csv_write(dest / "paired_validation.csv", paired)
    write_json(dest / "summary.json", {"completed_runs": len(final), "available_runs": len(available),
               "paired_validation": paired, "retention_label": c["retention_label"],
               "interpretation": "Positive mean_forgetting_reduction favors the intervention. "
               "Matched points linearly interpolate the first validation crossing, not actual held-out test checkpoints. "
               "CI resamples training seeds; few seeds give weak uncertainty estimates. No LR is selected using test results."})
    text = ["# Frozen-head CPT analysis", "", c["retention_label"], "",
            f"Completed runs: {len(final)}; trajectories available: {len(available)}.", "",
            "- `final_test.csv`: actual final held-out outcomes relative to the shared A anchor.",
            "- `trajectories.csv`: validation loss, adaptation, forgetting, head swaps, geometry, tokens, memory.",
            "- `matched_validation.csv`: first-crossing interpolation at the requested gain targets; unreached targets are explicit.",
            "- `paired_validation.csv`: full minus intervention forgetting, paired by seed and LR; positive favors intervention.", "",
            "Matched-adaptation results are exploratory validation estimates. Do not claim held-out confirmation from them. "
            "For a confirmatory run, choose LR and a gain target on validation, then use a new configured run with "
            "`stop_at_validation_gain` and inspect its end-only test results, including the actual achieved gain.", "",
            "A frozen head helps only if its lower forgetting is accompanied by useful new learning. "
            "Head swaps are interventions on potentially co-adapted components, not an additive causal attribution."]
    (dest / "README.md").write_text("\n".join(text) + "\n")
    if plots and trajectories:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        colors = dict(zip(c["modes"], plt.cm.tab10.colors))
        labeled = set()
        for (seed, lr, mode), rows in available.items():
            a0, b0 = rows[0]["a"]["loss"], rows[0]["b"]["loss"]
            f = [r["a"]["loss"] - a0 for r in rows]
            g = [b0 - r["b"]["loss"] for r in rows]
            t = [(r["train_a_tokens"] + r["train_b_tokens"]) / 1e6 for r in rows]
            label = mode if mode not in labeled else None
            for ax, x, y in ((axes[0], t, f), (axes[1], t, g), (axes[2], g, f)):
                ax.plot(x, y, color=colors[mode], alpha=0.55, label=label)
            labeled.add(mode)
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.axhline(0, color="gray", lw=0.7)
        axes[0].set(xlabel="Total training target tokens (millions)", ylabel="Old-domain loss increase (nats)", title="Forgetting")
        axes[1].set(xlabel="Total training target tokens (millions)", ylabel="New-domain loss reduction (nats)", title="Adaptation")
        axes[2].set(xlabel="New-domain loss reduction (nats)", ylabel="Old-domain loss increase (nats)", title="Retention–adaptation trade-off")
        axes[2].legend(fontsize=8)
        fig.suptitle("Validation trajectories · each line is one seed / learning rate")
        fig.tight_layout()
        fig.savefig(dest / "retention_adaptation.png", dpi=180)
        fig.savefig(dest / "retention_adaptation.svg")
        plt.close(fig)
    print(f"Analysis: {dest}")

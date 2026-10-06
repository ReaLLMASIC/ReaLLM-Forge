"""Summarize task learning separately from retention and benchmark forgetting."""
from pathlib import Path

from analyze import csv_write, load_rows
from common import read_json, run_dir, write_json


def first_accuracy_crossing(rows, target):
    # Accuracy is discrete and noisy. Select an ACTUAL checkpoint, never interpolate an accuracy.
    return next((r for r in rows if r["math"]["relaxed_accuracy"] >= target), None)


def report_task(c, plots=True):
    root = Path(c["output"]) / "task_analysis"
    root.mkdir(parents=True, exist_ok=True)
    baseline = read_json(Path(c["output"]) / "baseline" / "test.json")
    trajectories, final, matched = [], [], []
    for seed in c["seeds"]:
        for lr in c["lrs"]:
            for mode in c["modes"]:
                path = run_dir(c, seed, mode, lr)
                rows = load_rows(path / "metrics.jsonl")
                for r in rows:
                    trajectories.append({"seed": seed, "lr": lr, "mode": mode, "step": r["step"],
                       "math_accuracy": r["math"]["relaxed_accuracy"], "strict_math_accuracy": r["math"]["strict_accuracy"],
                       "format_rate": r["math"]["format_rate"], "math_gain_pp": r["math_gain_pp"],
                       "response_loss": r["math"]["response_loss"], "retention_loss_increase": r["retention_loss_increase"],
                       "response_tokens": r["response_tokens"], "task_examples": r["task_examples"]})
                initial_accuracy = rows[0]["math"]["relaxed_accuracy"] if rows else 0.0
                targets = c["sft"].get("accuracy_targets", [initial_accuracy + g for g in c["sft"].get("accuracy_gain_targets", [0.05, 0.1, 0.15])])
                for target in targets:
                    row = first_accuracy_crossing(rows, target)
                    matched.append({"seed": seed, "lr": lr, "mode": mode, "target_accuracy": target,
                       "target_gain_pp": 100 * (target - initial_accuracy),
                       "baseline_already_at_target": initial_accuracy >= target,
                       "reached": row is not None, "step": row["step"] if row else None,
                       "actual_accuracy": row["math"]["relaxed_accuracy"] if row else None,
                       "retention_loss_increase": row["retention_loss_increase"] if row else None})
                if (path / "complete.json").exists():
                    r = read_json(path / "test.json")
                    final.append({"seed": seed, "lr": lr, "mode": mode, "step": r["step"],
                       "math_test_accuracy": r["math"]["relaxed_accuracy"],
                       "strict_math_test_accuracy": r["math"]["strict_accuracy"],
                       "math_test_gain_pp": 100 * (r["math"]["relaxed_accuracy"] - baseline["math"]["relaxed_accuracy"]),
                       "math_test_examples": r["math"]["examples"], "format_rate": r["math"]["format_rate"],
                       "generation_cap_rate": r["math"]["generation_cap_rate"],
                       "old_test_loss_increase": r["retention"]["loss"] - baseline["retention"]["loss"]})
    csv_write(root / "validation_trajectories.csv", trajectories)
    csv_write(root / "final_test.csv", final)
    csv_write(root / "validation_accuracy_crossings.csv", matched)
    write_json(root / "summary.json", {"completed_runs": len(final), "baseline_test": baseline,
               "interpretation": "Positive math gain is useful new learning. Positive old loss increase is forgetting. "
               "Accuracy crossings select actual validation checkpoints with potentially different overshoots; "
               "they are exploratory, not held-out confirmation. Benchmark accuracy changes are in benchmarks/*/retention.csv."})
    if plots and trajectories:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        colors = dict(zip(c["modes"], plt.cm.tab10.colors))
        seen = set()
        for seed in c["seeds"]:
            for lr in c["lrs"]:
                for mode in c["modes"]:
                    rows = [r for r in trajectories if (r["seed"], r["lr"], r["mode"]) == (seed, lr, mode)]
                    if not rows:
                        continue
                    label = mode if mode not in seen else None
                    axes[0].plot([r["step"] for r in rows], [100*r["math_accuracy"] for r in rows], color=colors[mode], alpha=0.65, label=label)
                    axes[1].plot([100*r["math_accuracy"] for r in rows], [r["retention_loss_increase"] for r in rows], color=colors[mode], alpha=0.65, label=label)
                    seen.add(mode)
        axes[0].set(xlabel="Optimizer step", ylabel="GSM8K validation accuracy (%)", title="Task acquisition")
        axes[1].set(xlabel="GSM8K validation accuracy (%)", ylabel="WikiText loss increase (nats/token)", title="Retention at comparable math accuracy")
        for ax in axes:
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "task_retention.png", dpi=180)
        fig.savefig(root / "task_retention.svg")
        plt.close(fig)
    print(f"Task analysis: {root}")

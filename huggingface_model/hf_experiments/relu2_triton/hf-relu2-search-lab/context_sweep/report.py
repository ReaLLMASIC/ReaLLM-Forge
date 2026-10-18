"""Truthful live tables and exportable validation-versus-context figures."""
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time
from .recipe import ROOT, read, write, progress

COLORS = {"softmax_sdpa": "#2563eb", "relu2": "#0f766e", "linear16": "#9333ea", "kda": "#d97706"}
COLORS.update(relu2_bias="#c2410c", relu2_scale_half="#0891b2", relu2_scale_one="#65a30d",
              relu2_bias_scale_half="#db2777", relu2_bias_scale_one="#854d0e")
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon + AdamW"}
NORM_LABELS = {"none": "No output norm", "capped": "Capped output norm"}
LABELS = {"softmax_sdpa": "Softmax SDPA", "relu2": "ReLU²", "kda": "Kimi Delta Attention", **{f"linear{n}": f"Linear at {n}" for n in (2, 4, 8, 16)}}
LABELS.update(relu2_bias="ReLU² + bias", relu2_scale_half="ReLU² + α=½", relu2_scale_one="ReLU² + α=1",
              relu2_bias_scale_half="ReLU² + bias + α=½", relu2_bias_scale_one="ReLU² + bias + α=1")


def rows(path):
    result = []
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return result
    for line in text.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except json.JSONDecodeError:
            pass  # Atomic checkpoints remain the source of committed progress.
    return result


def best(validations, key, target):
    allowed = [row for row in validations if row.get("eligible_for_best") and
        0 < row.get("tokens", 0) <= target and isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
    return min(allowed, key=lambda row: row[key]) if allowed else None


def collect(root):
    root = Path(root)
    plan = read(root / "plan.json", {})
    state = read(root / "sweep_state.json", {})
    run_rows, validation_rows, norm_rows, attention_rows = [], [], [], []
    latest_jobs = {j["run"]: j for j in state.get("jobs", [])}
    initial = {}
    for cell in plan.get("cells", []):
        run = root / cell["name"]
        info = read(run / "run.json", {})
        records = rows(run / "metrics.jsonl")
        vals = [r for r in records if r.get("kind") == "validation"]
        train = [r for r in records if r.get("kind") == "train"]
        _, saved = progress(root, cell["name"])
        summary = read(run / "train_summary.json", {})
        tokens = max([r.get("tokens", 0) for r in records] + [saved.get("cursor", 0)])
        complete = summary.get("status") == "complete" and summary.get("tokens") == cell["target_tokens"] and saved.get("cursor") == cell["target_tokens"]
        selected_best = best(vals, "loss", cell["target_tokens"])
        common_best = best(vals, "common_loss", cell["target_tokens"])
        job = latest_jobs.get(cell["name"], {})
        active = (state.get("active") or {}).get("run") == cell["name"]
        status = "complete" if complete else "running" if active else "failed" if job.get("status") == "failed" else "paused" if tokens else "pending"
        steady = [r["tokens_per_second"] for r in train[-30:] if r.get("steady") and r.get("tokens_per_second", 0) > 0]
        if info.get("shared_initial_state_sha256"):
            family = "kda" if cell["variant"] == "kda" else "full_attention"
            initial.setdefault(f"{family}/seed{cell['seed']}", set()).add(info["shared_initial_state_sha256"])
        entry = dict(cell, status=status, tokens=tokens, saved_tokens=saved.get("cursor", 0),
            recent_tokens_per_second=statistics.median(steady) if steady else None,
            steady_tokens_per_second=summary.get("steady_tokens_per_second"),
            best_validation_loss=selected_best["loss"] if selected_best else None,
            best_validation_tokens=selected_best["tokens"] if selected_best else None,
            best_common_loss=common_best["common_loss"] if common_best else None,
            best_common_tokens=common_best["tokens"] if common_best else None,
            final_validation_loss=summary.get("validation", {}).get("loss") if complete else None,
            common_validation_context=plan["spec"]["common_validation_context"],
            completed_budget=complete, parameter_count=info.get("parameter_counts", {}).get("total"),
            best_eligible_checks=sum(bool(r.get("eligible_for_best")) for r in vals),
            log=str(root / "logs" / f"{cell['name']}.log"),
            last_error=job.get("error") if job.get("status") == "failed" else None)
        last_diagnostics = vals[-1].get("output_norm_diagnostics", []) if vals else []
        for branch in ("attention", "ffn"):
            ds = [d for d in last_diagnostics if d["branch"] == branch]
            count = sum(d["vectors"] for d in ds)
            entry[f"{branch}_fraction_above_radius"] = (sum(d["vectors"] * d["fraction_above_radius"] for d in ds) / count) if count else None
        run_rows.append(entry)
        for row in vals:
            identity = dict(run=cell["name"], variant=cell["variant"], optimizer=cell["optimizer"], output_norm=cell["output_norm"], series=cell["series"], context=cell["context"], seed=cell["seed"])
            validation_rows.append(dict(identity, **{k:v for k,v in row.items() if k not in ("output_norm_diagnostics", "attention_parameters")}))
            norm_rows.extend(dict(identity, step=row["step"], tokens=row["tokens"], **d) for d in row.get("output_norm_diagnostics", []))
            for parameter in row.get('attention_parameters', []):
                biases=parameter.get('subtractive_bias')
                for head,bias in enumerate(biases if biases is not None else [None]*plan['spec']['model']['num_attention_heads']):
                    attention_rows.append(dict(identity,step=row['step'],tokens=row['tokens'],head=head,
                        **{k:v for k,v in parameter.items() if k!='subtractive_bias'},subtractive_bias=bias))
    mismatched_seeds = [seed for seed, values in initial.items() if len(values) != 1]
    counts = {r["parameter_count"] for r in run_rows if r["parameter_count"] is not None}
    if not counts:
        checked = read(root / "preflight.json", {})
        counts = {v["total"] for v in checked.get("parameter_counts", {}).values()}
    paired = []
    for row in run_rows:
        if row["output_norm"] != "capped":
            continue
        baseline = next((r for r in run_rows if r["output_norm"] == "none" and all(r[k] == row[k] for k in ("variant", "context", "seed", "optimizer"))), None)
        ready = bool(baseline and row["completed_budget"] and baseline["completed_budget"] and not mismatched_seeds)
        paired.append(dict(variant=row["variant"], context=row["context"], seed=row["seed"], optimizer=row["optimizer"], complete_pair=ready,
            **{f"delta_{key}": row[key] - baseline[key] if ready and row[key] is not None and baseline[key] is not None else None
               for key in ("best_validation_loss", "best_common_loss", "final_validation_loss")}))
    return dict(updated_at=time.time(), plan=plan, state=state, runs=run_rows, validations=validation_rows, output_norm_diagnostics=norm_rows, attention_parameters=attention_rows, paired_deltas=paired,
        completed=sum(r["completed_budget"] for r in run_rows), total=len(run_rows),
        parameter_count=next(iter(counts)) if len(counts) == 1 else None, parameter_counts=sorted(counts),
        processed_training_tokens=sum(r["tokens"] for r in run_rows), initialization_mismatch_seeds=mismatched_seeds,
        note="Best scheduled validation within the fixed token budget; initial and pause-only checks are excluded. "
             "Filled points require the full budget for every planned seed at that point. "
            "Optimizer recipes are recorded per run. Solid: no output norm; dashed: capped attention and FFN output norms. "
             "Bias arms add one zero-initialized learned scalar per head/layer in auxiliary AdamW. "
             "KDA is a pure recurrent arm with no RoPE, extra gates/convolutions and its own always-on gated head RMSNorm; parameter counts differ. "
             "Native-context evaluation changes available context; the separate common-context plot uses identical evaluation windows.")


def atomic_csv(path, data):
    keys = list(dict.fromkeys(k for row in data for k in row))
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    os.replace(tmp, path)


def plot(live, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    spec = data["plan"]["spec"]
    smoke = spec.get("smoke_only", False)
    target = data["plan"]["tokens_per_run"]
    count = data.get("parameter_count")
    counts = data.get("parameter_counts", [])
    size = f"{count/1e6:.2f}M parameters • " if count is not None else (f"{min(counts)/1e6:.2f}–{max(counts)/1e6:.2f}M parameters • " if counts else "Matched backbone • ")
    prefix = "CPU functional smoke • synthetic data" if smoke else f"{size}{target/1e6:.3f}M training tokens/run"
    contexts = data["plan"]["contexts"]
    for metric, filename, title in (
        ("best_validation_loss", "best_validation_vs_context", "Best validation loss at each training context"),
        ("best_common_loss", "common_context_validation", f"Best validation loss at shared context {spec['common_validation_context']:,}"),
        ("final_validation_loss", "final_validation_vs_context", "Final validation loss at matched training tokens")):
        multi_optimizer=len(data['plan']['optimizers'])>1
        fig, ax = plt.subplots(figsize=(11, 10 if multi_optimizer else 7.5), layout="constrained")
        any_completed = False
        for series in data["plan"]["series"]:
            variant, optimizer = series["variant"], series["optimizer"]
            values, deviations = [], []
            partial_x, partial_y = [], []
            for context in contexts:
                members = [r for r in data["runs"] if r["series"] == series["key"] and r["context"] == context]
                completed = [r[metric] for r in members if r["completed_budget"] and r[metric] is not None]
                ready = len(completed) == len(data["plan"]["seeds"]) and not data["initialization_mismatch_seeds"]
                values.append(statistics.mean(completed) if ready else np.nan)
                deviations.append(statistics.stdev(completed) if ready and len(completed) > 1 else 0)
                any_completed |= ready
                if not ready:
                    for r in members:
                        if r[metric] is not None:
                            partial_x.append(context)
                            partial_y.append(r[metric])
            ax.errorbar(contexts, values, yerr=deviations if len(data["plan"]["seeds"]) > 1 else None,
                color=COLORS[variant], label=LABELS[variant] + " · " + NORM_LABELS[series["output_norm"]]+(' · '+OPTIMIZER_LABELS[optimizer] if multi_optimizer else ''),
                linestyle="--" if series["output_norm"] == "capped" else "-", marker=('D' if series['output_norm']=='capped' else '^') if optimizer=='adamw' else ('s' if series['output_norm']=='capped' else 'o'),
                markersize=5, linewidth=1.7, capsize=3)
            if partial_x:
                ax.scatter(partial_x, partial_y, facecolors="none", edgecolors=COLORS[variant], alpha=0.35, s=38, marker="s" if series["output_norm"] == "capped" else "o")
        ax.set_xscale("log", base=2)
        ax.set_xlim(min(contexts)/1.2,max(contexts)*1.2)
        ax.set_xticks(contexts, [f"{n:,}" for n in contexts])
        ax.set(xlabel="Training context length (tokens)", ylabel="Cross-entropy loss (nats/token; lower is better)", title=title)
        ax.grid(alpha=0.17)
        ax.legend(ncol=2, fontsize=8.5, loc="best")
        fig.suptitle(prefix, fontsize=12)
        ax.text(0, -0.19, "Filled: completed budget. Hollow: best so far, incomplete budget.\nMultiple seeds: mean ± sample SD; single-seed points have no uncertainty estimate.",
                transform=ax.transAxes, fontsize=8.5, color="#555555")
        if not any_completed:
            ax.text(0.5, 0.5, "No completed-budget points yet", transform=ax.transAxes, ha="center", color="#555555")
        for ext in ("png", "pdf", "svg"):
            path = live / f"{filename}.{ext}"
            tmp = path.with_name(f"{filename}.{os.getpid()}.tmp.{ext}")
            fig.savefig(tmp, dpi=170)
            os.replace(tmp, path)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    for variant,optimizer in ((v,o) for v in data['plan']['variants'] for o in data['plan']['optimizers']):
        values, deviations = [], []
        for context in contexts:
            members = [r["delta_best_validation_loss"] for r in data["paired_deltas"]
                       if r["variant"] == variant and r["optimizer"] == optimizer and r["context"] == context and r["complete_pair"]]
            ready = len(members) == len(data["plan"]["seeds"]) and all(v is not None for v in members)
            values.append(statistics.mean(members) if ready else np.nan)
            deviations.append(statistics.stdev(members) if ready and len(members) > 1 else 0)
        ax.errorbar(contexts, values, yerr=deviations if len(data["plan"]["seeds"]) > 1 else None,
                    label=LABELS[variant]+(' · '+OPTIMIZER_LABELS[optimizer] if len(data['plan']['optimizers'])>1 else ''), color=COLORS[variant], marker="o" if optimizer=='muon' else '^', linestyle='-' if optimizer=='muon' else '--', capsize=3)
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(contexts)/1.2,max(contexts)*1.2)
    ax.set_xticks(contexts, [f"{n:,}" for n in contexts])
    ax.set(xlabel="Training context length (tokens)", ylabel="Best native loss: capped − no output norm", title="Effect of the two output norms (negative is better)")
    ax.legend(); ax.grid(alpha=.17); fig.suptitle(prefix, fontsize=12)
    if not any(r['complete_pair'] for r in data['paired_deltas']):
        ax.text(.5,.65,'No completed pairs yet',transform=ax.transAxes,ha='center',color='#555555')
    ax.text(0, -.17, "Only paired completed budgets are shown. Single seed: no uncertainty estimate.", transform=ax.transAxes, fontsize=8.5, color="#555555")
    for ext in ("png", "pdf", "svg"):
        path = live / f"output_norm_delta.{ext}"
        tmp = path.with_name(f"output_norm_delta.{os.getpid()}.tmp.{ext}")
        fig.savefig(tmp, dpi=170); os.replace(tmp, path)
    plt.close(fig)


def refresh(root, plots=True):
    root = Path(root)
    live = root / "live"
    live.mkdir(parents=True, exist_ok=True)
    data = collect(root)
    if not data["plan"]:
        return data
    write(live / "live_data.json", data)
    atomic_csv(live / "sweep_results.csv", data["runs"])
    atomic_csv(live / "validation.csv", data["validations"])
    atomic_csv(live / "output_norm_diagnostics.csv", data["output_norm_diagnostics"])
    atomic_csv(live / "attention_parameters.csv", data["attention_parameters"])
    atomic_csv(live / "paired_deltas.csv", data["paired_deltas"])
    lines = ["# 50M attention context sweep", "", data["note"], "", f"Updated: {time.ctime(data['updated_at'])}",
        f"Completed: {data['completed']}/{data['total']}; target per run: {data['plan']['tokens_per_run']:,} tokens.", "",
        "The token budget is per run, with no wall-time limit. Compatible backbone and QKV/output-projection weights, Muon/AdamW recipe, optimizer-step tokens, data prefix and validation cadence are matched. KDA adds gates and convolution filters; exact parameter counts differ. Initialization is checked within each architecture family. The capped arm adds 2 × layers × width zero-initialized gain parameters. The affine gain acts on all vectors; only radial normalization is conditional.", "",
        "![Best validation vs context](best_validation_vs_context.png)", "",
        "![Shared-context validation](common_context_validation.png)", "",
        "![Final validation at matched tokens](final_validation_vs_context.png)", "",
        "[Checkpoint diagnostic graphs](diagnostics/index.html). Per-head bias, QK scale and length exponent history: `attention_parameters.csv`. Detailed probes: `diagnostic_attention.csv`, `diagnostic_loss_by_position.csv`, `diagnostic_branch_norms.csv`. Bias arms add one learned scalar per head/layer (51 at 17 layers × 3 heads).", "",
        "![Paired output-norm effect](output_norm_delta.png)", "",
        "| Context | Variant | Output norm | Parameters | Seed | Status | Tokens | Best native loss | Best at tokens | Best common loss | Recent tokens/s |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    def number(value, digits=4):
        return "—" if value is None else f"{value:,.{digits}f}"
    for r in data["runs"]:
        lines.append(f"| {r['context']} | {r['variant']} | {r['output_norm']} | {number(r['parameter_count'],0)} | {r['seed']} | {r['status']} | {r['tokens']:,} | {number(r['best_validation_loss'])} | {number(r['best_validation_tokens'], 0)} | {number(r['best_common_loss'])} | {number(r['recent_tokens_per_second'], 0)} |")
    if data["initialization_mismatch_seeds"]:
        lines += ["", "**Initialization mismatch:** " + ", ".join(data["initialization_mismatch_seeds"]) + ". Completed comparison curves are withheld."]
    lines += ["", "## Jobs needing attention", ""]
    for r in data["runs"]:
        if r["status"] == "failed":
            lines += [f"- {r['name']}: `{r['log']}`", "", "```text", r.get("last_error") or "See the worker log", "```", ""]
    if data["plan"]["spec"].get("smoke_only"):
        lines += ["", "CPU smoke results use tiny models and synthetic text; they are not evidence about the full-model sweep."]
    temp = live / f"report.{os.getpid()}.tmp"
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, live / "report.md")
    html = (ROOT / "context_sweep/dashboard.html").read_text()
    temp = live / f"index.{os.getpid()}.tmp"
    temp.write_text(html)
    os.replace(temp, live / "index.html")
    if plots:
        plot(live, data)
        from diagnostics.report import export
        export(root)
    return data

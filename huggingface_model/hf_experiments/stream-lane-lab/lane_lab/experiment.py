"""Small reproducible CPU/GPU experiments; explicit next-frame alignment."""
from __future__ import annotations
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .config import LaneConfig
from .model import LaneModel
from .data import sequence_bank


def means(model, logits):
    return torch.stack([(p.float().softmax(-1) * torch.linspace(
        c["minimum"], c["maximum"], c["levels"], device=p.device)).sum(-1)
        for p, c in zip(logits, model.config.columns)], -1)


def categorical_loss(model, logits, targets, target_mode="hard"):
    labels = model.quantize(targets)
    terms = []
    for c, (p, spec) in enumerate(zip(logits, model.config.columns)):
        if target_mode == "hard":
            terms.append(F.cross_entropy(p.reshape(-1, p.shape[-1]), labels[..., c].reshape(-1)))
        else:
            location = ((targets[..., c] - spec["minimum"]) /
                        (spec["maximum"] - spec["minimum"])).clamp(0, 1) * (spec["levels"] - 1)
            lo = location.floor().long()
            hi = (lo + 1).clamp_max(spec["levels"] - 1)
            fraction = location - lo
            logp = p.log_softmax(-1)
            terms.append(-((1 - fraction) * logp.gather(-1, lo[..., None]).squeeze(-1) +
                           fraction * logp.gather(-1, hi[..., None]).squeeze(-1)).mean())
    return torch.stack(terms).mean()


def evaluate_model(model, seed=30000, sequences=24, horizon=32, mode="mean"):
    """Unseen phase/amplitude seeds; future is used only after prediction."""
    model.eval()
    device = next(model.parameters()).device
    ctx = model.config.context_length
    bank = torch.from_numpy(sequence_bank(seed, sequences, ctx + horizon,
                                         len(model.config.columns))).to(device)
    context, truth = bank[:, :ctx], bank[:, ctx:]
    with torch.inference_mode():
        out = model(context, labels=model.quantize(bank[:, 1:ctx + 1]))
        teacher = means(model, out.logits)
        started = time.perf_counter()
        pred = model.forecast(context, horizon, mode=mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        persistence = context[:, -1:].expand_as(truth)
        linear = context[:, -1:] + (context[:, -1:] - context[:, -2:-1]) * torch.arange(
            1, horizon + 1, device=device)[None, :, None]
        rmse = lambda a, b: float((a - b).square().mean().sqrt())
        result = {
            "seed_start": seed, "sequences": sequences, "horizon": horizon, "mode": mode,
            "teacher_nll": float(out.cross_entropy),
            "teacher_rmse": rmse(teacher, bank[:, 1:ctx + 1]),
            "one_step_rmse": rmse(pred[:, :1], truth[:, :1]),
            "rollout_rmse": rmse(pred, truth),
            "persistence_rmse": rmse(persistence, truth),
            "linear_rmse": rmse(linear, truth),
            "persistence_one_step_rmse": rmse(persistence[:, :1], truth[:, :1]),
            "linear_one_step_rmse": rmse(linear[:, :1], truth[:, :1]),
            "rollout_rmse_per_column": (pred - truth).square().mean((0, 1)).sqrt().cpu().tolist(),
            "boundary_fraction": float(torch.stack([(((pred[..., i] - c["minimum"]) /
                (c["maximum"] - c["minimum"])).sub(.5).abs() >= .495)
                for i, c in enumerate(model.config.columns)]).float().mean()),
            "batched_forecast_ms": elapsed * 1000,
            "rmse_by_horizon": {str(h): rmse(pred[:, :h], truth[:, :h])
                                 for h in sorted(set([1, min(8, horizon), min(16, horizon), horizon]))},
            "persistence_rmse_by_horizon": {str(h): rmse(persistence[:, :h], truth[:, :h])
                                 for h in sorted(set([1, min(8, horizon), min(16, horizon), horizon]))},
        }
    return result, {"context": context[0].cpu().tolist(), "truth": truth[0].cpu().tolist(),
                    "prediction": pred[0].cpu().tolist()}


def load_config(args):
    if args.config:
        return LaneConfig(**json.loads(Path(args.config).read_text()))
    return LaneConfig(columns=[dict(name=f"signal_{i+1}", kind=args.variant,
        knots=args.knots, tied=args.tied, levels=args.levels, minimum=-1., maximum=1.,
        backend=args.backend) for i in range(args.channels)],
        d_model=args.width, n_heads=2, n_layers=args.layers, d_ff=args.width * 3,
        context_length=args.context, auxiliary_mse_weight=0.)


def initialize_matched_model(config, seed, device="cpu"):
    """Match backbone weights independently of column geometry parameter count.

    Construction is CPU-only. Restore the caller's CPU RNG afterwards; do not
    seed or consume CUDA generators while constructing CPU parameters.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = LaneModel(config)
        torch.random.default_generator.manual_seed(seed + 1)
        for module in (model.position, model.blocks, model.final_norm):
            module.apply(model._init_weights)
    return model.to(device)


def training_generators(seed, device="cpu"):
    """Independent data streams; geometry size and noise cannot alter batches."""
    batches = torch.Generator(device=device).manual_seed(seed + 10000)
    noise = torch.Generator(device=device).manual_seed(seed + 20000)
    return batches, noise


def train(args):
    output = Path(args.output)
    if (output / "train.jsonl").exists():
        raise ValueError(f"{output} already contains a run; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.init_model:
        if args.config:
            raise ValueError("Use --init-model or --config, not both")
        model = LaneModel.from_pretrained(args.init_model).to(args.device)
        config = model.config
    else:
        config = load_config(args)
        model = initialize_matched_model(config, args.seed, args.device)
    batch_generator, noise_generator = training_generators(args.seed, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    frames = config.context_length + max(1, args.rollout_steps)
    bank = torch.from_numpy(sequence_bank(1000, 512, frames, len(config.columns))).to(args.device)
    run = vars(args).copy()
    run.pop("func", None)
    run.update(config=config.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
               torch=torch.__version__, python=platform.python_version(),
               device=str(args.device), training_seeds=[1000, 1511], validation_seeds=[20000, 20007],
               test_seeds=[30000, 30023], selection_metric="validation rollout_rmse",
               initialization_protocol="checkpoint-finetune-v1" if args.init_model else "matched-v1",
               rng_seeds=dict(geometry=args.seed, backbone=args.seed + 1,
                              minibatches=args.seed + 10000, input_noise=args.seed + 20000),
               objective="CE + mse_weight * expected-value MSE + rollout_weight * rollout MSE")
    (output / "run.json").write_text(json.dumps(run, indent=2))
    best = math.inf
    started = time.perf_counter()
    log = (output / "train.jsonl").open("w")
    try:
        for step in range(1, args.steps + 1):
            model.train()
            ids = torch.randint(len(bank), (args.batch_size,), device=args.device,
                                generator=batch_generator)
            batch = bank[ids]
            inputs, targets = batch[:, :config.context_length], batch[:, 1:config.context_length+1]
            if args.input_noise:
                noise = torch.randn(inputs.shape, device=inputs.device, dtype=inputs.dtype,
                                    generator=noise_generator)
                inputs = (inputs + noise * args.input_noise).clamp(-1, 1)
            logits = model(inputs).logits
            ce = categorical_loss(model, logits, targets, args.target_mode)
            mse = F.mse_loss(means(model, logits), targets)
            loss = ce + args.mse_weight * mse
            rollout = torch.zeros((), device=args.device)
            # Predict each rollout point before looking at its target. Detached
            # generated inputs limit graph memory; each output retains its loss.
            if args.rollout_steps and args.rollout_weight and step > args.steps // 4 and step % args.rollout_every == 0:
                history = inputs
                terms = []
                for h in range(args.rollout_steps):
                    next_value = means(model, model(history[:, -config.context_length:]).logits)[:, -1:]
                    terms.append(F.mse_loss(next_value, batch[:, config.context_length+h:config.context_length+h+1]))
                    feedback = next_value if args.rollout_gradient == "full" else next_value.detach()
                    history = torch.cat([history, feedback], 1)
                rollout = torch.stack(terms).mean()
                loss = loss + args.rollout_weight * rollout
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            warmup = min(1., step / max(1, min(100, args.steps // 10)))
            lr = args.lr * warmup * (.15 + .85 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            record = dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()),
                          mse=float(mse.detach()), rollout_mse=float(rollout.detach()),
                          grad_norm=float(grad), lr=lr, elapsed_s=time.perf_counter()-started)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics, _ = evaluate_model(model, seed=20000, sequences=8, horizon=args.horizon)
                record["validation"] = metrics
                if metrics["rollout_rmse"] < best:
                    best = metrics["rollout_rmse"]
                    model.save_pretrained(output / "best")
                eta = record["elapsed_s"] / step * (args.steps-step)
                print(f"{output.name} {step}/{args.steps} CE={record['ce']:.4f} "
                      f"val_rollout={metrics['rollout_rmse']:.4f} best={best:.4f} ETA={eta:.0f}s", flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    finally:
        log.close()
    model.save_pretrained(output / "last")
    model = LaneModel.from_pretrained(output / "best").to(args.device)
    metrics, example = evaluate_model(model, horizon=args.horizon)
    metrics.update(parameters=run["parameters"], steps=args.steps,
                   training_seconds=time.perf_counter()-started,
                   cpu_threads=args.threads, cuda_verified=False)
    with torch.inference_mode():
        context = torch.tensor(example["context"], device=args.device)[None]
        model.forecast(context, args.horizon)  # warm-up outside timing
        timings = []
        for _ in range(3):
            if str(args.device).startswith("cuda"): torch.cuda.synchronize()
            before = time.perf_counter()
            model.forecast(context, args.horizon)
            if str(args.device).startswith("cuda"): torch.cuda.synchronize()
            timings.append((time.perf_counter()-before)*1000)
        metrics["single_stream_horizon_ms_median"] = float(np.median(timings))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "example.json").write_text(json.dumps(example))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def report(runs, output):
    import csv
    import html
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = [file for file in sorted(Path(runs).glob("**/metrics.json"))
                  if (file.parent / "train.jsonl").exists()]
    counts = sorted({len(json.loads((f.parent / "run.json").read_text())["config"]["columns"])
                     for f in candidates}) or [1]
    fig, axes = plt.subplots(len(counts), 2, figsize=(12, 3.8 * len(counts)),
                            layout="constrained", squeeze=False)
    for file in sorted(Path(runs).glob("**/metrics.json")):
        root = file.parent
        if not (root / "train.jsonl").exists(): continue
        m = json.loads(file.read_text())
        channels = len(json.loads((root / "run.json").read_text())["config"]["columns"])
        pair = axes[counts.index(channels)]
        name = str(root.relative_to(runs))
        rows.append(dict(run=name, **{k:m[k] for k in ["parameters", "teacher_nll", "one_step_rmse",
            "rollout_rmse", "persistence_rmse", "linear_rmse", "single_stream_horizon_ms_median"] if k in m}))
        rows[-1]["rmse_8"] = m.get("rmse_by_horizon", {}).get("8")
        rows[-1]["rmse_16"] = m.get("rmse_by_horizon", {}).get("16")
        logs = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
        ev = [x for x in logs if "validation" in x]
        pair[0].plot([x["step"] for x in ev], [x["validation"]["teacher_nll"] for x in ev], label=name)
        pair[1].plot([x["step"] for x in ev], [x["validation"]["rollout_rmse"] for x in ev], label=name)
    for channel, pair in zip(counts, axes):
        for ax, title in zip(pair, ["Teacher-forced validation NLL", "Free-running validation RMSE"]):
            ax.set(title=f"{channel} channel(s) · {title}", xlabel="Optimizer step within stage")
            ax.grid(alpha=.2)
            if rows: ax.legend(fontsize=7)
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (out / "comparison.csv").open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    header = "".join(f"<th>{html.escape(k)}</th>" for k in rows[0]) if rows else ""
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(round(v,5) if isinstance(v,float) else v))}</td>" for v in r.values()) + "</tr>" for r in rows)
    (out / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Stream Lane Lab results</title>
    <style>body{{font:15px system-ui;margin:40px;background:#f4f6fa;color:#17283a}}img{{max-width:100%}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px;border:1px solid #dde2eb;text-align:right}}th{{font-size:12px}}</style>
    <h1>Stream Lane Lab · measured results</h1><p>Best checkpoint selected on validation rollout RMSE. Table uses disjoint test seeds. One seed per configuration; not a statistical ranking.</p>
    <img src="curves.png" alt="Validation loss and free-running error curves"><div style="overflow:auto"><table><tr>{header}</tr>{body}</table></div>''')
    return out / "index.html"

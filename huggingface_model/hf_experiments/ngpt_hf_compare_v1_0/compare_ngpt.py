#!/usr/bin/env python3
"""Single-GPU Hugging Face nGPT comparison: prepare / train / plot / smoke.

Train both models FROM SCRATCH, on identical sampled next-token batches.
One iteration = one optimizer update, including all accumulated microbatches.
"""
import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from model import ModelConfig, Transformer
from data import prepare, load_tokens, synthetic_data, starts_for_step, token_batch

FIELDS = ["step", "tokens", "train_loss", "train_probe_loss", "val_loss", "lr", "grad_norm",
          "train_seconds", "elapsed_seconds", "step_seconds", "tokens_per_second",
          "peak_allocated_gib", "unit_norm_max_error", "alpha_attn_mean", "s_z_mean"]


def learning_rate(step, steps, peak, warmup):
    """step is 1-based. No hidden scheduler warmup; final update has LR zero."""
    if not 1 <= step <= steps or not 0 <= warmup < steps:
        raise ValueError("Invalid schedule bounds")
    if warmup and step <= warmup:
        return peak*step/warmup
    progress = (step-warmup)/(steps-warmup) if warmup else (step-1)/max(steps-1, 1)
    return peak*0.5*(1+math.cos(math.pi*progress))


def optimizer_for(model, variant, lr):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    # AdamW with wd=0 is Adam; moments themselves are NOT projected.
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": .1 if variant == "gpt" else 0.0},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=lr, betas=(.9, .95), eps=1e-8)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(args, device):
    return torch.autocast(device.type, dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()


def versions():
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "cuda_runtime": torch.version.cuda}
    for name in ("transformers", "datasets", "numpy", "matplotlib", "huggingface_hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def atomic_save(state, path):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, arrays, args, device):
    model.eval()
    results = {}
    for i, split in enumerate(("train", "val")):
        # Separate RNG, re-created each evaluation; never alters training samples.
        rng = np.random.default_rng(np.random.SeedSequence([args.data_seed, 98765, i]))
        loss_sum = 0.0
        for _ in range(args.eval_batches):
            starts = rng.integers(0, len(arrays[split])-args.context, size=args.batch_size)
            tokens = token_batch(arrays[split], starts, args.context, device)
            with autocast(args, device):
                loss_sum += model.next_token_loss(tokens, args.loss_chunk).item()
        results["train_probe_loss" if split == "train" else "val_loss"] = loss_sum/args.eval_batches
    if not all(math.isfinite(x) for x in results.values()):
        raise FloatingPointError(f"Non-finite evaluation: {results}")
    model.train()
    return results


def train_one(args, variant, seed, meta, arrays, device):
    c = ModelConfig(vocab_size=meta["vocab_size"], width=args.width, layers=args.layers,
                    heads=args.heads, context=args.context, variant=variant,
                    activation_checkpointing=args.activation_checkpointing,
                    weight_storage=args.weight_storage)
    peak = args.lr_gpt if variant == "gpt" else args.lr_ngpt
    warmup = args.warmup_gpt if variant == "gpt" else 0
    if warmup >= args.steps:
        raise ValueError("GPT warmup must be below total steps; use --warmup-gpt for short trials.")
    spec = dict(asdict(c), steps=args.steps, batch_size=args.batch_size,
                accumulation=args.accumulation, peak_lr=peak, warmup=warmup,
                precision=args.precision, grad_clip=args.grad_clip, seed=seed,
                data_seed=args.data_seed, data_sha256=meta["sha256"],
                eval_batches=args.eval_batches, eval_every=args.eval_every,
                loss_chunk=args.loss_chunk, core_only=args.core_only,
                deterministic=args.deterministic,
                weight_decay=.1 if variant == "gpt" else 0.0,
                adam_betas=[.9, .95], adam_eps=1e-8)
    out = Path(args.out)/f"{variant}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out/"last.pt"
    if (out/"run.json").exists() and not args.resume:
        raise FileExistsError(f"Existing run at {out}. Use --resume or a different --out.")
    if args.resume and (out/"run.json").exists():
        prior = json.loads((out/"run.json").read_text())["spec"]
        if prior != spec:
            changed = [k for k in set(prior)|set(spec) if prior.get(k) != spec.get(k)]
            raise ValueError(f"Resume would change the experiment: {changed}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Run started but has no completed checkpoint: {out}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if args.core_only:
        model = Transformer(c)
    else:
        from transformers import AutoModelForCausalLM
        from hf_model import NGPTConfig
        model = AutoModelForCausalLM.from_config(NGPTConfig(**asdict(c)))
    model = model.to(device)
    optimizer = optimizer_for(model, variant, peak)
    count = sum(p.numel() for p in model.parameters())
    # last.pt + temporary atomic replacement + final weights, with a safety margin.
    free_needed = count*28 + 1024**3
    if shutil.disk_usage(out).free < free_needed:
        raise OSError(f"Need approximately {free_needed/1024**3:.1f} GiB free for this run's checkpoints.")
    manifest = dict(spec=spec, parameter_count=count, backbone_matrix_parameter_count=sum(
        p.numel() for _, p, _ in (model if args.core_only else model.net).constrained_weights()),
        synthetic=meta.get("synthetic", False), versions=versions(),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        training_tokens_per_update=args.context*args.batch_size*args.accumulation,
        dataset_metadata=meta)
    history, start, train_seconds, elapsed_before = [], 0, 0.0, 0.0
    if args.resume and ckpt.exists():
        # Load only trusted checkpoints produced by this script. weights_only=True
        # limits deserialization to tensors and basic containers.
        saved = torch.load(ckpt, map_location=device, weights_only=True)
        if saved["spec"] != spec:
            raise ValueError("Checkpoint experiment specification does not match.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history = saved["step"], saved["history"]
        train_seconds, elapsed_before = saved["train_seconds"], saved["elapsed_seconds"]
        torch.set_rng_state(saved["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        del saved
    (out/"run.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begun = time.perf_counter()
    def elapsed():
        return elapsed_before + time.perf_counter()-begun
    def save(step):
        sync(device)
        atomic_save(dict(spec=spec, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         step=step, history=history, train_seconds=train_seconds,
                         elapsed_seconds=elapsed(), rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), ckpt)
    model.train()
    print(f"\n{out.name}: {count:,} parameters; {args.context*args.batch_size*args.accumulation:,} tokens/update; "
          f"lr={peak:g}, warmup={warmup}, storage={args.weight_storage}", flush=True)
    if not history:
        row = {"step": 0, "tokens": 0, "train_seconds": 0.0, "lr": 0.0}
        row.update(evaluate(model, arrays, args, device))
        row.update(model.diagnostics())
        row["elapsed_seconds"] = elapsed()
        history.append(row)
        print(f"step 0 | train probe {row['train_probe_loss']:.5f} | val {row['val_loss']:.5f}", flush=True)
    write_csv(out/"metrics.csv", history)  # discard rows beyond last durable checkpoint on resume
    stop = min(args.steps, args.stop_after_steps or args.steps)
    if stop < start:
        raise ValueError("--stop-after-steps is earlier than the saved checkpoint.")
    with (out/"metrics.csv").open("a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, FIELDS)
        for step in range(start+1, stop+1):
            sync(device)
            t0 = time.perf_counter()
            lr = learning_rate(step, args.steps, peak, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for micro in range(args.accumulation):
                starts = starts_for_step(len(arrays["train"]), args.context, args.batch_size,
                                         args.data_seed, step, micro)
                tokens = token_batch(arrays["train"], starts, args.context, device)
                with autocast(args, device):
                    loss = model.next_token_loss(tokens, args.loss_chunk)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss at {variant} step {step}")
                train_loss += loss.detach().item()/args.accumulation
                (loss/args.accumulation).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                args.grad_clip if args.grad_clip > 0 else float("inf"), error_if_nonfinite=True).item()
            optimizer.step()
            # Once per COMPLETE update, not once per microbatch and not before step().
            model.project_weights_()
            sync(device)
            dt = time.perf_counter()-t0
            train_seconds += dt
            tpu = args.batch_size*args.accumulation*args.context
            row = dict(step=step, tokens=step*tpu, train_loss=train_loss,
                       lr=lr, grad_norm=grad_norm, train_seconds=train_seconds,
                       step_seconds=dt, tokens_per_second=tpu/dt,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.0)
            # Deliberately do not add off-cadence evaluation when pausing: an
            # interrupted/resumed run has exactly the same metric rows as a full run.
            if step % args.eval_every == 0 or step == args.steps:
                row.update(evaluate(model, arrays, args, device))
                row.update(model.diagnostics())
                print(f"step {step}/{args.steps} | train {train_loss:.5f} | "
                      f"val {row['val_loss']:.5f} | {tpu/dt:,.0f} tok/s | "
                      f"ETA {(args.steps-step)*(train_seconds/step)/3600:.2f}h + eval/checkpoints", flush=True)
            elif step % args.log_every == 0:
                print(f"step {step}/{args.steps} | train {train_loss:.5f} | lr {lr:.3g} | {tpu/dt:,.0f} tok/s", flush=True)
            row["elapsed_seconds"] = elapsed()
            history.append(row)
            writer.writerow(row)
            if step % args.save_every == 0 or step == stop:
                save(step)
    if stop == args.steps and not args.core_only:
        model.save_pretrained(out/"final_hf", safe_serialization=True)
        tokdir = Path(args.data)/"tokenizer"
        if tokdir.exists():
            shutil.copytree(tokdir, out/"final_hf", dirs_exist_ok=True)
    vals = [r for r in history if "val_loss" in r]
    summary = {"completed_steps": stop, "parameter_count": count,
               "best_recorded_val_loss": min(r["val_loss"] for r in vals),
               "latest_recorded_val_loss": vals[-1]["val_loss"],
               "train_seconds": train_seconds, "elapsed_seconds": elapsed(),
               "note": "Synthetic plumbing test only" if meta.get("synthetic") else "Single experiment; not a claim of paper-level replication"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train(args):
    device = torch.device(args.device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This simple runner is single-process/single-GPU; do not use torchrun.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the PyTorch installation and GPU driver.")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16. Use --precision fp32 for both models.")
    if args.weight_storage == "reference_bf16" and args.precision != "bf16":
        raise ValueError("reference_bf16 storage requires --precision bf16.")
    for key in ("steps", "batch_size", "accumulation", "eval_every", "eval_batches", "save_every", "log_every", "loss_chunk"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_','-')} must be positive.")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(args.deterministic)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    meta, arrays = load_tokens(args.data, args.context)
    for seed in args.seeds:
        for variant in args.variants:
            train_one(args, variant, seed, meta, arrays, device)
    from plot_losses import plot_runs
    plot_runs(args.out, args.smooth)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("prepare", help="Stream bounded HF data to reusable token files")
    d.add_argument("--data", default="data/owt_200m")
    d.add_argument("--dataset", default="Skylion007/openwebtext")
    d.add_argument("--dataset-config", default=None)
    d.add_argument("--dataset-revision", default="main")
    d.add_argument("--train-split", default="train")
    d.add_argument("--validation-split", default=None)
    d.add_argument("--text-column", default="text")
    d.add_argument("--tokenizer", default="gpt2")
    d.add_argument("--tokenizer-revision", default="main")
    d.add_argument("--train-tokens", type=int, default=200_000_000)
    d.add_argument("--val-tokens", type=int, default=1_000_000)
    d.add_argument("--data-seed", type=int, default=1234)
    d.add_argument("--shuffle-buffer", type=int, default=1024)
    t = sub.add_parser("train", help="Train the matched comparison, then plot")
    t.add_argument("--data", default="data/owt_200m")
    t.add_argument("--out", default="runs/owt_4090")
    t.add_argument("--variants", nargs="+", choices=["gpt", "ngpt"], default=["gpt", "ngpt"])
    t.add_argument("--seeds", nargs="+", type=int, default=[0])
    for name, default in [("width",512), ("layers",8), ("heads",8), ("context",1024),
                          ("steps",10000), ("batch-size",2), ("accumulation",16),
                          ("warmup-gpt",2000), ("eval-every",100), ("eval-batches",16),
                          ("save-every",500), ("log-every",10), ("loss-chunk",128),
                          ("data-seed",1234), ("cpu-threads",4), ("smooth",1)]:
        t.add_argument(f"--{name}", type=int, default=default)
    t.add_argument("--lr-gpt", type=float, default=.003)
    t.add_argument("--lr-ngpt", type=float, default=.003)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    t.add_argument("--weight-storage", choices=["fp32", "reference_bf16"], default="fp32")
    t.add_argument("--activation-checkpointing", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stop-after-steps", type=int, default=None)
    t.add_argument("--core-only", action="store_true", help="Offline PyTorch testing; NOT HF integration")
    t.add_argument("--deterministic", action="store_true")
    q = sub.add_parser("plot")
    q.add_argument("--out", default="runs/owt_4090")
    q.add_argument("--smooth", type=int, default=1)
    s = sub.add_parser("smoke", help="Offline synthetic CPU plumbing test; not a benchmark")
    s.add_argument("--out", default="runs/synthetic_smoke")
    s.add_argument("--steps", type=int, default=100)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "plot":
        from plot_losses import plot_runs
        plot_runs(args.out, args.smooth)
    else:
        data_path = str(Path(args.out)/"synthetic_data")
        synthetic_data(data_path)
        args = p.parse_args(["train", "--data", data_path, "--out", args.out,
            "--device", "cpu", "--precision", "fp32", "--core-only", "--width", "64",
            "--layers", "2", "--heads", "4", "--context", "32", "--batch-size", "4",
            "--accumulation", "1", "--steps", str(args.steps), "--warmup-gpt", "10",
            "--eval-every", "10", "--eval-batches", "2", "--save-every", "50",
            "--loss-chunk", "128", "--smooth", "5", "--cpu-threads", "2"])
        train(args)


if __name__ == "__main__":
    main()

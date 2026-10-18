"""Resumable matched training, with actual full-step wall-clock timing."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from .common import (read_json, write_json, digest, json_digest, state_digest, environment,
                     source_hashes, amp, sync, configure_device, make_model, parameter_counts,
                     budget, validate_config, TokenStream, token_loss, backend_labels)
from .optimizers import optimizer_for, OPTIMIZERS
from hf_model.output_norm import start_output_diagnostics, stop_output_diagnostics


def learning_rate(step, total_steps, cfg, base_lr):
    warm = cfg["warmup_steps"]
    if step < warm: return base_lr * (step + 1) / max(1, warm)
    progress = (step - warm) / max(1, total_steps - warm - 1)
    return base_lr * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(1, progress))))


@torch.no_grad()
def validate(model, stream, cfg, device, dtype, length=None):
    model.eval()
    losses = []
    n = length or cfg["sequence_length"]
    micro_tokens = cfg["micro_batch_size"] * cfg["sequence_length"]
    b = max(1, micro_tokens // n)
    evaluated_tokens = cfg.get("validation_tokens", micro_tokens * cfg["eval_batches"])
    if evaluated_tokens % (b * n):
        raise ValueError("Validation tokens must divide into identical complete batches")
    for i in range(evaluated_tokens // (b * n)):
        x, y = stream.batch(i * b * n, b, n, device)
        with amp(device, dtype): losses.append(float(token_loss(model, x, y, cfg["loss_chunk_tokens"])))
    model.train()
    loss = float(np.mean(losses))
    if not math.isfinite(loss): raise FloatingPointError("Nonfinite validation loss")
    return dict(loss=loss, perplexity=math.exp(loss) if loss < 700 else None,
                evaluated_tokens=evaluated_tokens, context_length=n)


def validation_suite(model, stream, cfg, device, dtype):
    start_output_diagnostics(model)
    try:
        native = validate(model, stream, cfg, device, dtype)
    finally:
        diagnostics = stop_output_diagnostics(model)
    common_length = cfg.get("common_validation_context", cfg["sequence_length"])
    common = native if common_length == cfg["sequence_length"] else validate(model, stream, cfg, device, dtype, common_length)
    attention_parameters = []
    for index, layer in enumerate(model.model.layers):
        bias = getattr(layer.attn, "subtractive_bias", None)
        scale = getattr(layer.attn, "qk_norm_factor", None)
        attention_parameters.append(dict(layer=index,
            scale=float(scale.detach()) if scale is not None else None,
            subtractive_bias=bias.detach().float().cpu().tolist() if bias is not None else None,
            length_alpha=model.config.causal_length_alpha, length_anchor=model.config.causal_length_anchor))
    return dict(native, output_norm_diagnostics=diagnostics, attention_parameters=attention_parameters, common_loss=common["loss"], common_context_length=common_length,
                common_evaluated_tokens=common["evaluated_tokens"])


def update_best(state, evaluation, step, tokens, eligible):
    """Select only the shared validation cadence and final budget, never pause-only checks."""
    if not eligible:
        return
    for field, key in (("loss", "best_validation"), ("common_loss", "best_common_validation")):
        previous = state.get(key)
        if previous is None or evaluation[field] < previous["loss"]:
            state[key] = dict(loss=evaluation[field], step=step, tokens=tokens,
                context_length=evaluation["context_length" if field == "loss" else "common_context_length"])


def checkpoint_save(out, model, optimizer, data_dir, state, keep=1):
    final = out / f"checkpoint-{state['step']:08d}"
    temp = out / f".checkpoint-{state['step']:08d}.tmp"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir()
    model.save_pretrained(temp, safe_serialization=True)
    shutil.copytree(Path(data_dir) / "tokenizer", temp, dirs_exist_ok=True)
    payload = dict(state, optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(payload, temp / "training_state.pt")
    write_json(temp / "progress.json", {k: v for k, v in state.items() if k != "step_times"})
    if final.exists(): shutil.rmtree(final)
    os.replace(temp, final)
    write_json(out / "latest.json", dict(checkpoint=final.name, step=state["step"]))
    old = sorted(out.glob("checkpoint-[0-9]*"))
    for path in old[:-keep]: shutil.rmtree(path)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, help="Stop at this absolute step while preserving the original LR schedule")
    a = p.parse_args()
    cfg = validate_config(read_json(a.config))
    t = cfg["train"]
    if a.device == "cpu" and not cfg.get("smoke_only"):
        p.error("Full-size runs require a CUDA GPU; CPU is reserved for the smoke config")
    configure_device(a.device, cfg["dtype"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    train, val = TokenStream(a.data, "train"), TokenStream(a.data, "validation")
    if train.manifest["max_token_id"] >= cfg["model"]["vocab_size"]: raise ValueError("Tokenizer exceeds model vocabulary")
    per_step, total_steps, total_tokens = budget(cfg)
    if len(train.tokens) < total_tokens + 1: raise ValueError(f"Prepare at least {total_tokens + 1} training tokens")
    if len(val.tokens) <= t["micro_batch_size"] * t["sequence_length"] * t["eval_batches"]:
        raise ValueError("Not enough validation tokens")
    # Detect corruption/replacement once at startup, outside measured steps.
    for split in ("train", "validation"):
        if digest(Path(a.data) / f"{split}.bin") != train.manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} stream hash changed")
    for name, sha in train.manifest["tokenizer_files"].items():
        if digest(Path(a.data) / "tokenizer" / name) != sha:
            raise ValueError(f"Prepared tokenizer changed: {name}")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()) and not a.resume: p.error("Output already exists; use --resume or a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(config=cfg, variant=a.variant, optimizer=a.optimizer, seed=a.seed, data_manifest_sha256=digest(Path(a.data) / "manifest.json"),
                    source_hashes=source_hashes(), environment=environment())
    signature = json_digest(identity)
    model, initial_hash = make_model(cfg, a.variant, a.seed, a.device)
    shared_initial_hash = state_digest(model, shared_only=True)
    model.model.gradient_checkpointing = t["gradient_checkpointing"]
    model.train()
    opt = optimizer_for(model, a.optimizer, cfg["optimizer_recipes"][a.optimizer], a.device)
    state = dict(signature=signature, step=0, cursor=0, step_times=[], steady_tokens=0,
                 steady_seconds=0.0, train_step_seconds=0.0, elapsed_seconds=0.0,
                 initial_state_sha256=initial_hash, last_validation=None)
    if a.resume and (out / "latest.json").exists():
        ckpt = out / read_json(out / "latest.json")["checkpoint"]
        saved = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
        if saved["signature"] != signature: raise ValueError("Config/data/code/environment changed. Resume requires the same experiment.")
        from hf_model.modeling_comparison import ComparisonForCausalLM
        restored = ComparisonForCausalLM.from_pretrained(ckpt)
        model.load_state_dict(restored.state_dict(), strict=True)
        del restored
        opt.load_state_dict(saved.pop("optimizer"))
        torch.set_rng_state(saved.pop("torch_rng"))
        np.random.set_state(saved.pop("numpy_rng"))
        random.setstate(saved.pop("python_rng"))
        cuda_rng = saved.pop("cuda_rng")
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        state = saved
        # Drop uncommitted log entries after the most recent atomic checkpoint.
        log = out / "metrics.jsonl"
        if log.exists():
            lines, kept = log.read_text().splitlines(), []
            for index, line in enumerate(lines):
                try:
                    if json.loads(line)["step"] <= state["step"]: kept.append(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1: raise
            log.write_text("\n".join(kept) + ("\n" if kept else ""))
    elif a.resume and any(out.iterdir()):
        raise ValueError("No complete checkpoint found in the existing output")
    write_json(out / "optimizer_groups.json", opt.assignment)
    write_json(out / "run.json", dict(identity, signature=signature, parameter_counts=parameter_counts(model),
               optimizer_assignment=opt.assignment,
               output_norm=model.config.output_norm, shared_initial_state_sha256=shared_initial_hash,
               initial_state_sha256=initial_hash, requested_tokens=t["token_budget"], actual_token_budget=total_tokens,
               tokens_per_step=per_step, planned_steps=total_steps))
    if state["step"] >= total_steps:
        print("Training complete; verifying final export and summary.")
    stopped = False
    def stop_handler(signum, frame):
        nonlocal stopped
        stopped = True
        print("Stopping after the current optimizer step and saving a checkpoint.", flush=True)
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    start_step = state["step"]
    session_start = time.perf_counter()
    prior_elapsed = state["elapsed_seconds"]
    if str(a.device).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    with open(out / "metrics.jsonl", "a", buffering=1) as log:
        def record(row):
            log.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if state["step"] == 0:
            initial_eval = validation_suite(model, val, t, a.device, cfg["dtype"])
            record(dict(kind="validation", step=0, tokens=0, eligible_for_best=False,
                        elapsed_seconds=time.perf_counter() - session_start, **initial_eval))
        while state["step"] < total_steps:
            step = state["step"]
            for group in opt.param_groups:
                group["lr"] = learning_rate(step, total_steps, t, group["initial_lr"])
            rates = {group["group_name"]: group["lr"] for group in opt.param_groups}
            lr = opt.param_groups[0]["lr"]
            sync(a.device)
            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=a.device)
            for micro in range(t["gradient_accumulation"]):
                cursor = state["cursor"] + micro * t["micro_batch_size"] * t["sequence_length"]
                x, y = train.batch(cursor, t["micro_batch_size"], t["sequence_length"], a.device)
                with amp(a.device, cfg["dtype"]):
                    loss = token_loss(model, x, y, t["loss_chunk_tokens"])
                (loss / t["gradient_accumulation"]).backward()
                loss_sum += loss.detach() / t["gradient_accumulation"]
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"], error_if_nonfinite=True)
            opt.step()
            sync(a.device)
            seconds = time.perf_counter() - started
            loss_value = float(loss_sum)
            if not math.isfinite(loss_value): raise FloatingPointError("Training diverged; latest completed checkpoint was retained")
            state["step"] += 1
            state["cursor"] += per_step
            state["train_step_seconds"] += seconds
            steady = step - start_step >= t["timing_warmup_steps"]
            if steady:
                state["steady_tokens"] += per_step
                state["steady_seconds"] += seconds
                state["step_times"].append(seconds)
            state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - session_start
            if state["step"] % t["log_every"] == 0 or state["step"] == 1:
                record(dict(kind="train", step=state["step"], tokens=state["cursor"], loss=loss_value,
                       lr=lr, learning_rates=rates, optimizer=a.optimizer, output_norm=model.config.output_norm, grad_norm=float(grad_norm), step_seconds=seconds, tokens_per_second=per_step / seconds,
                       steady=steady, elapsed_seconds=state["elapsed_seconds"], backends=backend_labels(model)))
            final_step = state["step"] == total_steps
            stop_now = stopped or (a.stop_after is not None and state["step"] >= a.stop_after)
            if state["step"] % t["eval_every"] == 0 or final_step or stop_now:
                evaluation = validation_suite(model, val, t, a.device, cfg["dtype"])
                state["last_validation"] = evaluation
                eligible = state["step"] % t["eval_every"] == 0 or final_step
                update_best(state, evaluation, state["step"], state["cursor"], eligible)
                state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - session_start
                record(dict(kind="validation", step=state["step"], tokens=state["cursor"],
                            eligible_for_best=eligible, elapsed_seconds=state["elapsed_seconds"], **evaluation))
            if state["step"] % t["save_every"] == 0 or final_step or stop_now:
                state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - session_start
                checkpoint_save(out, model, opt, a.data, state, keep=t.get("keep_checkpoints", 1))
            if stop_now: break
    state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - session_start
    if state["step"] == total_steps:
        model.save_pretrained(out / "final", safe_serialization=True)
        shutil.copytree(Path(a.data) / "tokenizer", out / "final", dirs_exist_ok=True)
    summary = dict(status="complete" if state["step"] == total_steps else "paused", variant=a.variant, optimizer=a.optimizer, seed=a.seed,
                   output_norm=model.config.output_norm, shared_initial_state_sha256=shared_initial_hash,
                   parameter_counts=parameter_counts(model), tokens=state["cursor"], steps=state["step"],
                   initial_state_sha256=initial_hash, final_state_sha256=state_digest(model),
                   validation=state["last_validation"], backends=backend_labels(model),
                   best_validation=state.get("best_validation"), best_common_validation=state.get("best_common_validation"),
                   steady_tokens_per_second=state["steady_tokens"] / state["steady_seconds"] if state["steady_seconds"] else None,
                   steady_step_median_seconds=float(np.median(state["step_times"])) if state["step_times"] else None,
                   train_step_seconds=state["train_step_seconds"], elapsed_seconds_excluding_final_export=state["elapsed_seconds"],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(a.device).startswith("cuda") else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(a.device).startswith("cuda") else None,
                   note="Full optimizer steps; includes data transfer and LM-head recomputation; excludes evaluation/checkpointing from steady timing. CPU smoke is not a performance result.")
    write_json(out / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

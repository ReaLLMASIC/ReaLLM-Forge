"""Project-only disk controls and atomic, complete full-weight checkpoints."""
from pathlib import Path
from dataclasses import asdict
import importlib.metadata
import json
import os
import random
import shutil
import time
import torch
import numpy as np
from safetensors.torch import save_model, load_model


def project_root():
    return Path(os.environ.get("SLERP_WORK_ROOT", Path.cwd())).resolve()


def directory_bytes(path):
    # Avoid double counting hard-linked snapshots and following symlinks.
    seen, total = set(), 0
    for folder, _, files in os.walk(path, followlinks=False):
        for name in files:
            p = Path(folder)/name
            if p.is_symlink():
                continue
            st = p.stat(); key = (st.st_dev, st.st_ino)
            if key not in seen:
                seen.add(key); total += st.st_size
    return total


def disk_guard(cfg, additional_bytes=0):
    root = project_root()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free-additional_bytes < cfg.min_free_gb * 1e9:
        raise RuntimeError(f"Disk guard: need {cfg.min_free_gb:g} GB free after write; available {free/1e9:.2f} GB")
    # .venv, model cache, data and runs all count toward this project cap.
    used = directory_bytes(root)
    if used+additional_bytes > cfg.max_project_gb*1e9:
        raise RuntimeError(f"Disk guard: project uses {used/1e9:.2f} GB, cap {cfg.max_project_gb:g} GB")
    return {"project_bytes":used,"free_bytes":free}


def inventory(model):
    return [{"name":n,"shape":list(p.shape),"count":p.numel(),"dtype":str(p.dtype)}
            for n,p in model.named_parameters() if p.requires_grad]


def versions():
    result = {}
    for p in ["torch", "transformers", "accelerate", "datasets", "numpy", "bitsandbytes"]:
        try: result[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: result[p] = None
    return result


def trainable_state(model):
    """Legacy API name; v0.2 returns ALL unique model parameters."""
    return {n:p.detach().cpu().contiguous() for n,p in model.named_parameters()}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(v) for v in value)
    return 0


def save_weights(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.cfg.save(path/"config.json")
    model.original.config.save_pretrained(path/"backbone_config")
    # Preserve generation stop IDs as well as the architecture configuration.
    model.original.generation_config.save_pretrained(path/"backbone_config")
    # safetensors handles tied input/output embeddings and validates them on load.
    save_model(model, str(path/"model.safetensors"))
    if not model.cfg.tiny:
        from .data import tokenizer_for
        tokenizer_for(model.cfg).save_pretrained(path/"tokenizer")
    (path/"format.json").write_text(json.dumps({"version":"0.3.0", "training_mode":"full",
        "unique_parameters":sum(p.numel() for p in model.parameters()),
        "backbone_parameters":sum(p.numel() for p in model.backbone.parameters()),
        "full_backbone_saved":True}, indent=2)+"\n")


def checkpoint_path(path):
    path = Path(path)
    if (path/"latest.json").exists():
        return path/json.loads((path/"latest.json").read_text())["directory"]
    return path


def save_checkpoint(model, optimizer, run_dir, progress, scaler=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Space for an entire new checkpoint is required BEFORE the old one is removed.
    estimate = sum(p.numel()*p.element_size() for p in model.parameters())
    estimate += tensor_bytes(optimizer.state_dict()) + 100_000_000
    disk_guard(model.cfg, estimate)
    name = f"step-{progress['step']:08d}"
    dest = run_dir/name
    if dest.exists():
        raise FileExistsError(f"Checkpoint already exists: {dest}")
    tmp = run_dir/(name+".pending")
    if tmp.exists():
        raise FileExistsError(f"Incomplete checkpoint exists: {tmp}; inspect/remove it before retrying")
    tmp.mkdir()
    save_weights(model, tmp)
    # Tensor/primitive-only training state is safe to load with weights_only=True.
    training = {"optimizer":optimizer.state_dict(),"progress":progress,
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng":random.getstate(),"numpy_rng":list(np.random.get_state()[:1]) +
            [np.random.get_state()[1].tolist()] + list(np.random.get_state()[2:]),
        "versions":versions(), "scaler":scaler.state_dict() if scaler else None}
    torch.save(training, tmp/"training.pt")
    (tmp/"COMPLETE").write_text("0.3.0\n")
    for file in tmp.rglob("*"):
        if file.is_file():
            with file.open("rb") as fh: os.fsync(fh.fileno())
    os.replace(tmp,dest)
    pointer = run_dir/"latest.pending.json"
    pointer.write_text(json.dumps({"directory":name})+"\n")
    os.replace(pointer, run_dir/"latest.json")
    # Keep only the configured number after the complete replacement is committed.
    completed = sorted(p for p in run_dir.glob("step-*") if p.is_dir() and (p/"COMPLETE").exists())
    for old in completed[:-model.cfg.keep_checkpoints]:
        shutil.rmtree(old)
    return dest


def load_weights(model, path):
    path = checkpoint_path(path)
    if not (path/"model.safetensors").exists():
        raise ValueError("Need a full-weight checkpoint; v0.1 adapters cannot be resumed")
    load_model(model, str(path/"model.safetensors"), strict=True, device="cpu")
    return path


def finalize_run(run_dir):
    """Explicitly drop resumability after evaluation; retain full weights in place."""
    run_dir = Path(run_dir)
    path = checkpoint_path(run_dir)
    if not (path/"COMPLETE").exists() or not (path/"model.safetensors").exists():
        raise ValueError("Choose a completed full-weight run")
    training = path/"training.pt"
    removed = training.stat().st_size if training.exists() else 0
    if training.exists(): training.unlink()
    (path/"WEIGHTS_ONLY").write_text("Optimizer removed by explicit finalize command; start a new stage to train again.\n")
    return {"checkpoint":str(path), "removed_optimizer_bytes":removed, "resumable":False}


def append_json(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, allow_nan=False)+"\n")


def archive_run(run_dir, destination, mbps=20):
    """Manual sequential transfer; reading model/cache is never part of archiving."""
    import tarfile
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if not run_dir.is_dir() or not (run_dir/"latest.json").exists():
        raise ValueError("Choose a completed run directory")
    if destination.is_relative_to(run_dir):
        raise ValueError("Archive destination must be outside the run")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(run_dir)
    files = sorted(p for p in run_dir.iterdir() if p.is_file())
    files += sorted(p for p in checkpoint.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    if shutil.disk_usage(destination).free < size+1_000_000_000:
        raise RuntimeError("Archive destination has insufficient space")
    target = destination/(run_dir.name+"-"+time.strftime("%Y%m%dT%H%M%S")+".tar")
    if target.exists():
        raise FileExistsError(target)
    class LimitedWriter:
        def __init__(self, fh):
            self.fh, self.count, self.start = fh, 0, time.monotonic()
        def write(self, data):
            n=self.fh.write(data); self.count+=n
            wait=self.count/(mbps*1_000_000)-(time.monotonic()-self.start)
            if wait>0: time.sleep(min(wait,1))
            return n
        def flush(self): self.fh.flush()
    if mbps<=0: raise ValueError("mbps must be positive")
    with target.open("xb") as raw:
        with tarfile.open(fileobj=LimitedWriter(raw), mode="w|") as archive:
            for p in files:
                archive.add(p, arcname=str(Path(run_dir.name)/p.relative_to(run_dir)), recursive=False)
        raw.flush(); os.fsync(raw.fileno())
    return target

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from hf_model.configuration_comparison import ComparisonConfig, VARIANTS
from hf_model.modeling_comparison import ComparisonForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_digest(model, shared_only=False):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if shared_only and name.endswith((".attn_output_norm.gain", ".ffn_output_norm.gain", ".attn.subtractive_bias")):
            continue
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def environment():
    packages = {}
    for name in ("torch", "triton", "transformers", "tokenizers", "datasets", "lm_eval", "numpy", "fla-core", "einops"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for folder in ("hf_model", "experiment")
            for p in sorted((ROOT / folder).glob("*.py"))}


def amp(device, dtype):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and dtype == "bfloat16" else nullcontext()


def sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize()


def configure_device(device, dtype):
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; use --device cpu only with the smoke config")
        torch.cuda.set_device(torch.device(device))
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 not supported")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_model(cfg, variant, seed, device):
    seed_all(seed)
    args = dict(cfg["model"], variant=variant, linear_threshold=cfg["linear_threshold"])
    args["sdpa_backend"] = cfg["sdpa_backend"] if str(device).startswith("cuda") else "math"
    args["require_fused"] = str(device).startswith("cuda")
    args["relu2max_accelerator"] = "triton" if str(device).startswith("cuda") else "torch"
    model = ComparisonForCausalLM(ComparisonConfig(**args))
    if variant == "kda":
        # Copy matching backbone/QKV/O weights from the unchanged old arm.
        # Extra KDA parameters are initialized independently of the norm toggle.
        # A fork preserves the training RNG and never allocates a second GPU model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            baseline = ComparisonForCausalLM(ComparisonConfig(**dict(args, variant="relu2")))
        common = baseline.state_dict()
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in common and value.shape == common[name].shape:
                    value.copy_(common[name])
        del common, baseline
    initial_hash = state_digest(model)
    return model.to(device), initial_hash


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    embedding = model.get_input_embeddings().weight.numel()
    return dict(total=total, embedding=embedding, non_embedding=total - embedding)


def budget(cfg):
    t = cfg["train"]
    per_step = t["sequence_length"] * t["micro_batch_size"] * t["gradient_accumulation"]
    steps = math.ceil(t["token_budget"] / per_step)
    return per_step, steps, per_step * steps


def validate_config(cfg):
    t = cfg["train"]
    for key in ("sequence_length", "micro_batch_size", "gradient_accumulation", "token_budget", "eval_every", "save_every", "eval_batches", "log_every", "loss_chunk_tokens"):
        if not isinstance(t[key], int) or t[key] <= 0: raise ValueError(f"train.{key} must be a positive integer")
    if t["sequence_length"] > cfg["model"]["max_position_embeddings"]: raise ValueError("context exceeds configured maximum")
    if not set(cfg["variants"]) <= set(VARIANTS): raise ValueError("invalid variants")
    if t["warmup_steps"] < 0 or t["timing_warmup_steps"] < 0: raise ValueError("warmup must be nonnegative")
    if not 0 <= t["min_lr_ratio"] <= 1: raise ValueError("invalid LR multiplier")
    from .optimizers import validate_recipe
    for name in cfg["optimizers"]:
        validate_recipe(name, cfg["optimizer_recipes"][name])
    if cfg["dtype"] not in ("float32", "bfloat16"): raise ValueError("unsupported dtype")
    if t.get("keep_checkpoints", 1) < 1:
        raise ValueError("At least one full resume checkpoint must be retained")
    if not 1 <= t.get("common_validation_context", t["sequence_length"]) <= t["sequence_length"]:
        raise ValueError("Common validation context must be within every run's training context")
    return cfg


class TokenStream:
    """One packed stream, with exactly T next-token targets per T inputs.

    Training is sequential over the prepared, shuffled document stream; it never
    wraps silently. The saved cursor is enough for exact sample-order resume.
    """
    def __init__(self, directory, split):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        self.tokens = np.memmap(self.directory / f"{split}.bin", mode="r", dtype=np.dtype(self.manifest["dtype"]))
        expected = self.manifest["splits"][split]["tokens"]
        if len(self.tokens) != expected: raise ValueError(f"{split} size disagrees with manifest")

    def batch(self, cursor, batch_size, length, device):
        count = batch_size * length
        if cursor + count + 1 > len(self.tokens): raise ValueError("Token stream exhausted; prepare more data instead of repeating it")
        x = np.array(self.tokens[cursor:cursor + count], dtype=np.int64).reshape(batch_size, length)
        y = np.array(self.tokens[cursor + 1:cursor + count + 1], dtype=np.int64).reshape(batch_size, length)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if str(device).startswith("cuda"):
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def token_loss(model, x, y, chunk_tokens):
    """Chunk and recompute the LM head during training to bound vocabulary memory.

    Every variant uses identical code, and loss includes exactly x.numel() labels.
    Full-model training timings include this recomputation and the optimizer.
    """
    hidden = model.model(x, use_cache=False).last_hidden_state.reshape(-1, model.config.hidden_size)
    labels = y.reshape(-1)
    def ce(h, target):
        return F.cross_entropy(model.lm_head(h).float(), target, reduction="sum")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), chunk_tokens):
        h, target = hidden[start:start + chunk_tokens], labels[start:start + chunk_tokens]
        total = total + (checkpoint(ce, h, target, use_reentrant=False) if torch.is_grad_enabled() else ce(h, target))
    return total / len(labels)


def backend_labels(model): return sorted({layer.attn.last_backend for layer in model.model.layers})


def load_checkpoint_model(path, device, dtype, sdpa_backend=None):
    model = ComparisonForCausalLM.from_pretrained(path).to(device)
    model.config.sdpa_backend = sdpa_backend or ("flash" if str(device).startswith("cuda") else "math")
    model.config.require_fused = str(device).startswith("cuda")
    model.config.relu2max_accelerator = "triton" if str(device).startswith("cuda") else "torch"
    if dtype == "bfloat16" and str(device).startswith("cuda"):
        model.to(torch.bfloat16)
    return model.eval()

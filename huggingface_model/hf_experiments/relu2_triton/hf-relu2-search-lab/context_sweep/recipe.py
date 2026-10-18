"""Pure-Python sweep planning; planning does not initialize CUDA."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("softmax_sdpa", "relu2", "linear16", "kda", "relu2_bias", "relu2_scale_half",
            "relu2_scale_one", "relu2_bias_scale_half", "relu2_bias_scale_one")
OPTIMIZERS = ("adamw", "muon")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(spec):
    contexts, variants, seeds = spec["contexts"], spec["variants"], spec["seeds"]
    optimizers = spec["optimizers"]
    output_norms = spec.get("output_norms", ["none", "capped"])
    if not output_norms or len(set(output_norms)) != len(output_norms) or not set(output_norms) <= {"none", "capped"}:
        raise ValueError("Choose unique output norms from none and capped")
    if not optimizers or len(set(optimizers)) != len(optimizers) or not set(optimizers) <= set(OPTIMIZERS):
        raise ValueError("Choose unique optimizers from adamw and muon")
    if set(spec["optimizer_recipes"]) != set(optimizers):
        raise ValueError("Provide one optimizer recipe for each selected optimizer")
    if not contexts or len(set(contexts)) != len(contexts) or any(not isinstance(n, int) or n < 2 or n & (n - 1) for n in contexts):
        raise ValueError("Contexts must be unique powers of two >= 2")
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError(f"Choose unique variants from {VARIANTS}")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    for key in ("tokens_per_step", "requested_tokens_per_run", "microbatch_token_target", "max_micro_batch_size", "validation_tokens", "common_validation_context"):
        if not isinstance(spec[key], int) or isinstance(spec[key], bool) or spec[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    per = spec["tokens_per_step"]
    common = spec["common_validation_context"]
    if common > min(contexts) or max(contexts) > spec["model"]["max_position_embeddings"]:
        raise ValueError("Common context must fit every training context; model maximum must cover the entire sweep")
    steps = math.ceil(spec["requested_tokens_per_run"] / per)
    milestones = sorted({min(steps, math.ceil(t / per)) for t in spec["milestone_tokens"]} | {steps})
    if any(n < 1 for n in milestones):
        raise ValueError("Milestone tokens must be positive")
    configs, cells = {}, []
    for context in sorted(contexts):
        batch = max(1, min(spec["max_micro_batch_size"], spec["microbatch_token_target"] // context))
        micro = batch * context
        common_micro = max(1, micro // common) * common
        if per % micro or spec["validation_tokens"] % micro or spec["validation_tokens"] % common_micro:
            raise ValueError(f"Context {context}: optimizer/validation token counts must divide into full batches")
        train = dict(spec["train"], sequence_length=context, micro_batch_size=batch,
            gradient_accumulation=per // micro, token_budget=spec["requested_tokens_per_run"],
            eval_batches=spec["validation_tokens"] // micro, validation_tokens=spec["validation_tokens"],
            common_validation_context=common)
        base_config = dict(description=spec["name"], dtype=spec["dtype"], sdpa_backend=spec["sdpa_backend"],
            linear_threshold=16.0, optimizers=optimizers, optimizer_recipes=copy.deepcopy(spec["optimizer_recipes"]), cpu_threads=spec.get("cpu_threads", 4), variants=variants,
            model=copy.deepcopy(spec["model"]), train=train, smoke_only=spec.get("smoke_only", False))
        for output_norm in output_norms:
            config_key = f"{context}_{output_norm}"
            configs[config_key] = copy.deepcopy(base_config)
            configs[config_key]["model"]["output_norm"] = output_norm
            for seed in seeds:
                for variant in variants:
                    for optimizer in optimizers:
                        cells.append(dict(name=f"ctx{context}_{variant}_{optimizer}_norm-{output_norm}_seed{seed}", context=context,
                            variant=variant, optimizer=optimizer, output_norm=output_norm, config_key=config_key,
                            series=f"{variant}_{optimizer}_{output_norm}", seed=seed, steps=steps, target_tokens=steps * per,
                            batch_size=batch, accumulation=per // micro))
    return dict(spec=spec, recipe_sha256=fingerprint(spec), configs=configs, cells=cells,
        contexts=sorted(contexts), variants=variants, optimizers=optimizers, output_norms=output_norms,
        series=[dict(key=f"{v}_{o}_{n}", variant=v, optimizer=o, output_norm=n) for v in variants for o in optimizers for n in output_norms],
        seeds=seeds, milestones=milestones,
        tokens_per_step=per, steps=steps, tokens_per_run=steps * per,
        total_training_tokens=steps * per * len(cells), run_count=len(cells))


def progress(root, name):
    run = Path(root) / name
    latest = read(run / "latest.json", {})
    cp = run / latest.get("checkpoint", "missing")
    return cp, read(cp / "progress.json", {})


def scheduled_cells(plan, contexts=None, variants=None, seeds=None, round_index=0, optimizers=None, output_norms=None):
    chosen_contexts = set(contexts if contexts is not None else plan["contexts"])
    chosen_variants = set(variants if variants is not None else plan["variants"])
    chosen_seeds = set(seeds if seeds is not None else plan["seeds"])
    chosen_optimizers = set(optimizers if optimizers is not None else plan["optimizers"])
    chosen_norms = set(output_norms if output_norms is not None else plan["output_norms"])
    if not chosen_contexts <= set(plan["contexts"]) or not chosen_variants <= set(plan["variants"]) or not chosen_seeds <= set(plan["seeds"]) or not chosen_optimizers <= set(plan["optimizers"]) or not chosen_norms <= set(plan["output_norms"]):
        raise ValueError("Selections must be present in the preset; edit a copied preset for a different experiment")
    result = []
    for index, context in enumerate(plan["contexts"]):
        if context not in chosen_contexts:
            continue
        order = plan["series"]
        shift = (index + round_index) % len(order)
        order = order[shift:] + order[:shift]
        for seed in plan["seeds"]:
            if seed in chosen_seeds:
                for series in order:
                    if series["variant"] in chosen_variants and series["optimizer"] in chosen_optimizers and series["output_norm"] in chosen_norms:
                        result.append(next(c for c in plan["cells"] if c["context"] == context and c["seed"] == seed and c["series"] == series["key"]))
    return result

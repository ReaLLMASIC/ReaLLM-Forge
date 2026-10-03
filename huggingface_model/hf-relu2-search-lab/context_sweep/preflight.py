"""Check the supported environment, data and equal parameter counts."""
import importlib.metadata
import sys
from pathlib import Path
from .recipe import read, digest


def check(plan, data, device, selected_variants=None):
    for name, required in {"transformers": "4.44.2", "tokenizers": "0.19.1"}.items():
        installed = importlib.metadata.version(name)
        if installed != required:
            raise RuntimeError(f"{sys.executable}: {name}=={installed}; run bash scripts/setup.sh to install {name}=={required} in this package's venv")
    import numpy as np
    import torch
    from experiment.common import configure_device, environment, parameter_counts, source_hashes
    from hf_model.configuration_comparison import ComparisonConfig
    from hf_model.modeling_comparison import ComparisonForCausalLM
    from experiment.optimizers import describe_partition
    if "muon" in plan["optimizers"] and not hasattr(torch.optim, "Muon"):
        raise RuntimeError(f"{sys.executable}: PyTorch {torch.__version__} has no torch.optim.Muon; run bash scripts/setup.sh. No optimizer fallback is allowed.")
    if device == "cpu" and not plan["spec"].get("smoke_only"):
        raise ValueError("CPU is reserved for the functional smoke preset")
    configure_device(device, plan["spec"]["dtype"])
    if device.startswith("cuda"):
        import triton
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("Fused kernels require NVIDIA SM80 or newer")
        if "kda" in (selected_variants if selected_variants is not None else plan["variants"]):
            from hf_model.kda_attention import triton_ops
            triton_ops()
    data = Path(data).resolve()
    manifest = read(data / "manifest.json")
    if not manifest:
        raise ValueError(f"Missing prepared-data manifest: {data}")
    for split in ("train", "validation"):
        if (data / f"{split}.bin").stat().st_size != np.dtype(manifest["dtype"]).itemsize * manifest["splits"][split]["tokens"]:
            raise ValueError(f"{split}.bin does not match its manifest size")
    if manifest["splits"]["train"]["tokens"] < plan["tokens_per_run"] + 1:
        raise ValueError(f"Prepare at least {plan['tokens_per_run'] + 1:,} train tokens")
    if manifest["splits"]["validation"]["tokens"] < plan["spec"]["validation_tokens"] + 1:
        raise ValueError("Not enough validation tokens")
    if manifest["max_token_id"] >= plan["spec"]["model"]["vocab_size"]:
        raise ValueError("Prepared tokenizer exceeds the model vocabulary")
    for name, expected in manifest["tokenizer_files"].items():
        if digest(data / "tokenizer" / name) != expected:
            raise ValueError(f"Tokenizer changed: {name}")
    counts, assignments = {}, {}
    for output_norm in plan["output_norms"]:
        condition_counts = []
        for variant in plan["variants"]:
            args = dict(plan["spec"]["model"], variant=variant, output_norm=output_norm)
            with torch.device("meta"):
                model = ComparisonForCausalLM(ComparisonConfig(**args))
            key = f"{variant}_{output_norm}"
            counts[key] = parameter_counts(model)
            if variant != "kda":
                extra_bias = sum(p.numel() for name, p in model.named_parameters() if name.endswith(".attn.subtractive_bias"))
                expected_bias = args["num_hidden_layers"] * args["num_attention_heads"] if model.config.learned_subtractive_bias else 0
                if extra_bias != expected_bias:
                    raise RuntimeError("Unexpected subtractive-bias parameter count")
                condition_counts.append(counts[key]["total"] - extra_bias)
            assignments[key] = {name: describe_partition(model, name, plan["spec"]["optimizer_recipes"][name])
                                    for name in plan["optimizers"]}
            if model.lm_head.weight is not model.model.embed_tokens.weight:
                raise RuntimeError("Embedding weights must be tied")
        if len(set(condition_counts)) > 1:
            raise RuntimeError("Backbone counts differ after accounting for the per-head biases")
    if set(plan["output_norms"]) == {"none", "capped"}:
        expected_extra = 2 * plan["spec"]["model"]["num_hidden_layers"] * plan["spec"]["model"]["hidden_size"]
        for variant in plan["variants"]:
            if counts[f"{variant}_capped"]["total"] - counts[f"{variant}_none"]["total"] != expected_extra:
                raise RuntimeError("Unexpected extra parameters in the capped norm arm")
    return dict(status="ok", python=sys.executable, environment=environment(), parameter_counts=counts,
        optimizer_assignments=assignments, time_limit=None,
        data=str(data), data_manifest_sha256=digest(data / "manifest.json"), source_hashes=source_hashes(),
        note="KDA has extra gate/convolution parameters and no RoPE; this is a matched-backbone/token comparison, not equal parameter counts. The trainer also checks token-stream hashes. This check does not establish CUDA correctness or performance.")

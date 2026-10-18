"""Audited Muon/AdamW parameter routing using native PyTorch optimizers."""
import copy
import math
import torch
from torch import nn

OPTIMIZERS = ("adamw", "muon")


def validate_recipe(name, recipe):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    positive = ["learning_rate", "eps"] + (["aux_learning_rate", "aux_eps"] if name == "muon" else [])
    nonnegative = ["weight_decay"] + (["aux_weight_decay"] if name == "muon" else [])
    for key in positive + nonnegative:
        value = recipe[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or key in positive and value == 0:
            raise ValueError(f"Invalid {name}.{key}")
    betas = recipe["betas" if name == "adamw" else "aux_betas"]
    if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
        raise ValueError("AdamW betas must be in [0, 1)")
    if name == "muon":
        if not 0 <= recipe["momentum"] < 1 or not isinstance(recipe["nesterov"], bool):
            raise ValueError("Invalid Muon momentum/Nesterov settings")
        if not isinstance(recipe["ns_steps"], int) or recipe["ns_steps"] < 1:
            raise ValueError("Muon ns_steps must be positive")
        if len(recipe["ns_coefficients"]) != 3 or not all(math.isfinite(n) for n in recipe["ns_coefficients"]):
            raise ValueError("Muon needs three finite NS coefficients")
        if recipe["adjust_lr_fn"] not in ("original", "match_rms_adamw"):
            raise ValueError("Unsupported Muon LR adjustment")
    return recipe


def partition(model, name):
    """Select by actual module ownership and tensor identity, including tied weights."""
    if name not in OPTIMIZERS:
        raise ValueError(name)
    excluded = {id(p) for module in (model.get_input_embeddings(), model.get_output_embeddings())
                for p in module.parameters()}
    # Decoder Linear matrices and KDA depthwise filters represented as [D,W]
    # matrices enter Muon. Norms, gains, biases and decay scalars do not.
    decoder_weights = {id(module.weight) for module in model.model.layers.modules()
                       if isinstance(module, nn.Linear) or getattr(module, "_muon_matrix", False)}
    buckets = {"muon_decoder": [], "adamw_decay": [], "adamw_no_decay": []}
    seen = set()
    for parameter_name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            raise RuntimeError("A trainable tensor was assigned twice")
        seen.add(id(p))
        use_muon = name == "muon" and id(p) in decoder_weights and id(p) not in excluded and p.ndim == 2
        bucket = "muon_decoder" if use_muon else "adamw_decay" if p.ndim >= 2 else "adamw_no_decay"
        buckets[bucket].append((parameter_name, p))
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer partition does not cover all trainable parameters")
    if name == "muon" and not buckets["muon_decoder"]:
        raise ValueError("Muon selected but no eligible decoder matrices were found")
    return buckets


def describe_partition(model, name, recipe):
    validate_recipe(name, recipe)
    buckets = partition(model, name)
    groups = []
    for bucket, pairs in buckets.items():
        if not pairs:
            continue
        muon = bucket == "muon_decoder"
        lr = recipe["learning_rate"] if muon or name == "adamw" else recipe["aux_learning_rate"]
        decay = recipe["weight_decay"] if muon or name == "adamw" else recipe["aux_weight_decay"]
        if bucket == "adamw_no_decay":
            decay = 0.0
        groups.append(dict(name=bucket, implementation="torch.optim.Muon" if muon else "torch.optim.AdamW",
            tensor_count=len(pairs), parameter_count=sum(p.numel() for _, p in pairs),
            initial_lr=lr, weight_decay=decay,
            parameters=[dict(name=n, shape=list(p.shape), numel=p.numel()) for n, p in pairs]))
    return dict(optimizer=name, recipe=copy.deepcopy(recipe), groups=groups,
                total_parameters=sum(g["parameter_count"] for g in groups),
                tied_embeddings=model.get_input_embeddings().weight is model.get_output_embeddings().weight)


class OptimizerBundle:
    """Serialize both optimizers together; retain references to their live groups."""
    def __init__(self, name, recipe, optimizers, assignment):
        self.name, self.recipe = name, copy.deepcopy(recipe)
        self.optimizers, self.assignment = optimizers, assignment

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return dict(format_version=1, optimizer=self.name, recipe=self.recipe,
            assignment=self.assignment, states={name: opt.state_dict() for name, opt in self.optimizers.items()})

    def load_state_dict(self, saved):
        if (saved.get("format_version") != 1 or saved["optimizer"] != self.name or
                saved["recipe"] != self.recipe or saved["assignment"] != self.assignment or
                set(saved["states"]) != set(self.optimizers)):
            raise ValueError("Optimizer recipe or parameter assignment changed; cannot resume")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(saved["states"][name])


def optimizer_for(model, name, recipe, device):
    info = describe_partition(model, name, recipe)
    buckets = partition(model, name)
    if name == "muon" and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("This experiment requires torch.optim.Muon; use a PyTorch build that provides it (your 2.14 build does). No fallback optimizer is used.")
    optimizers = {}
    adam_groups = []
    for row in info["groups"]:
        bucket = row["name"]
        group = dict(params=[p for _, p in buckets[bucket]], param_names=[n for n, _ in buckets[bucket]],
            group_name=bucket, lr=row["initial_lr"], initial_lr=row["initial_lr"], weight_decay=row["weight_decay"])
        if bucket == "muon_decoder":
            optimizers["muon"] = torch.optim.Muon([group], lr=recipe["learning_rate"],
                momentum=recipe["momentum"], nesterov=recipe["nesterov"], ns_steps=recipe["ns_steps"],
                ns_coefficients=tuple(recipe["ns_coefficients"]), eps=recipe["eps"],
                adjust_lr_fn=recipe["adjust_lr_fn"])
        else:
            adam_groups.append(group)
    betas = recipe["betas" if name == "adamw" else "aux_betas"]
    eps = recipe["eps" if name == "adamw" else "aux_eps"]
    optimizers["adamw"] = torch.optim.AdamW(adam_groups, betas=tuple(betas), eps=eps,
                                           fused=str(device).startswith("cuda"))
    return OptimizerBundle(name, recipe, optimizers, info)

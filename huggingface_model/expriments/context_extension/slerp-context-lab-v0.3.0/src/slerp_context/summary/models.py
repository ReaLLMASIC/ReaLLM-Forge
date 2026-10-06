from dataclasses import asdict
from pathlib import Path
import torch
from ..config import Config
from ..model import RecurrentLM, build_base
from ..rmt import RMTLM
from ..storage import checkpoint_path, load_weights
from .config import file_digest


def device_for(requested=None):
    if requested=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_cfg(study,meta,method=None):
    cfg=study.base()
    if cfg.revision not in {"main",meta["model_revision"]}:
        raise ValueError("Requested model revision differs from prepared tokenizer")
    cfg.revision=meta["model_revision"]
    if method:cfg.method=method
    return cfg.validate()


def wrap(base,cfg):
    return RMTLM(base,cfg) if cfg.method=="rmt" else RecurrentLM(base,cfg)


def restore(path,device="cpu"):
    p=checkpoint_path(path);cfg=Config.load(p/"config.json")
    model=RecurrentLM.build(cfg,device,p/"backbone_config")
    load_weights(model,p)
    return model,p


def construct(cfg,device,initialize=None):
    if initialize:
        initial,_=restore(initialize,"cpu")
        if (initial.cfg.model_id,initial.cfg.revision)!=(cfg.model_id,cfg.revision):
            raise ValueError("Shared initialization model/revision differs")
        if initial.cfg.method!="native":
            raise ValueError("Shared initialization must be the native summarization SFT run")
        torch.manual_seed(cfg.seed)
        model=wrap(initial.backbone,cfg)
    else:model=RecurrentLM.build(cfg,"cpu")
    return model.to(device)


def weight_identity(model,path=None):
    return {"model_id":model.cfg.model_id,"revision":model.cfg.revision,
        "checkpoint_sha256":file_digest(checkpoint_path(path)/"model.safetensors") if path else None,
        "checkpoint_path":str(path) if path else None,
        "backbone_parameters":sum(p.numel() for p in model.backbone.parameters()),
        "backbone_weight_bytes":sum(p.numel()*p.element_size() for p in model.backbone.parameters()),
        "extra_parameters":sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.backbone.parameters())}

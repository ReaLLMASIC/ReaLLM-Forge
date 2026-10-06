from dataclasses import replace
from pathlib import Path
import json
import torch
from .model import RecurrentLM
from .data import tokenizer_for


@torch.no_grad()
def native_parity(cfg, output, device=None, length=64):
    """Compare pretrained forward passes before any eviction or training."""
    if length < 2 or length > cfg.window: raise ValueError("Parity length must fit the local window")
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    cfg=replace(cfg,method="local")
    model=RecurrentLM.build(cfg,device).eval()
    tok=tokenizer_for(cfg)
    text=tok.encode("A small model reads records carefully. The blue door is beside the garden. ",add_special_tokens=False)
    ids=torch.tensor([(text*((length+len(text)-1)//len(text)))[:length]],device=device)
    # FP32 reference isolates implementation parity from BF16 rounding differences.
    native=model.original(ids,use_cache=False).logits
    actual,_=model.consume(ids)
    delta=(native-actual).abs()
    passed=torch.allclose(native,actual,atol=2e-4,rtol=2e-4)
    result={"model_id":cfg.model_id,"revision":cfg.revision,"architecture":model.arch,
        "length":length,"dtype":"float32","max_absolute_error":float(delta.max()),
        "mean_absolute_error":float(delta.mean()),"passed":passed,
        "backbone_parameters":sum(p.numel() for p in model.backbone.parameters()),
        "all_backbone_trainable":all(p.requires_grad for p in model.backbone.parameters())}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not passed: raise RuntimeError("Native parity failed; do not start a long training run")
    return result

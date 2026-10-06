from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import time
import torch
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from .config import Config
from .model import RecurrentLM, build_base
from .data import tokenizer_for, training_episode, TokenDocuments
from .storage import (append_json, checkpoint_path, disk_guard, inventory, load_weights,
                      save_checkpoint, versions)


def accelerator_for(device=None):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; check the NVIDIA driver and PyTorch wheel")
    use_cpu = device == "cpu" or not torch.cuda.is_available()
    acc = Accelerator(cpu=use_cpu, mixed_precision="no" if use_cpu else "bf16")
    if acc.num_processes != 1:
        raise ValueError("One GPU/process only")
    return acc


def optimizer_for(model):
    if not all(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Full-weight training requires every backbone parameter to be trainable")
    groups = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = model.cfg.memory_lr if name.startswith("memories.") else model.cfg.backbone_lr
        wd = model.cfg.weight_decay if p.ndim>=2 and "bias" not in name else 0.0
        groups.setdefault((lr,wd),[]).append(p)
    groups = [{"params":p,"lr":lr,"base_lr":lr,"weight_decay":wd} for (lr,wd),p in groups.items()]
    if model.cfg.optimizer == "adamw8bit":
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("adamw8bit is supported on CUDA only in this package")
        try: from bitsandbytes.optim import AdamW8bit
        except ImportError as exc: raise RuntimeError("Install optional bitsandbytes==0.48.2") from exc
        return AdamW8bit(groups)
    return torch.optim.AdamW(groups, foreach=False)


def reference_for(cfg, device):
    if not cfg.kl_weight: return None
    teacher = build_base(cfg).requires_grad_(False).eval()
    return teacher.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)


def retention_kl(model, reference, ids):
    """Exact vocabulary KL(reference || student), mean over selected positions.

    This supervised retention term uses a short shared prefix, not RL rewards.
    No teacher probability mass is dropped by a top-k approximation.
    """
    x = ids[:, :model.cfg.kl_context]
    n = min(model.cfg.kl_tokens, x.shape[1])
    with torch.no_grad():
        transformer = reference.gpt_neox if reference.config.model_type == "gpt_neox" else reference.model
        h = transformer(x, use_cache=False).last_hidden_state[:, -n:]
        target = reference.get_output_embeddings()(h).float().log_softmax(-1)
    student = model.tail_logits(x, n).float().log_softmax(-1)
    return F.kl_div(student.reshape(-1, student.shape[-1]), target.reshape(-1, target.shape[-1]),
                    log_target=True, reduction="batchmean")


def train(cfg, run_dir, resume=False, initialize=None, device=None, max_updates=None):
    cfg.validate()
    if cfg.method == "native":
        raise ValueError("native is an unmodified inference baseline; use local/slerp/nlerp for training")
    run_dir = Path(run_dir)
    if resume and initialize:
        raise ValueError("Use resume OR initialize, not both")
    if resume:
        saved = Config.load(checkpoint_path(run_dir)/"config.json")
        # Changes to architecture, data, or budget during resume change the experiment.
        requested = asdict(cfg); previous = asdict(saved)
        requested["revision"] = previous["revision"]
        if json.dumps(requested, sort_keys=True) != json.dumps(previous, sort_keys=True):
            raise ValueError("Resume config differs. Use --initialize into a new run for a new stage")
        cfg = saved
        marker=run_dir/"TRAINING_COMPLETE.json"
        if marker.exists():
            completed=json.loads(marker.read_text())
            if completed["processed_tokens"] >= cfg.token_budget:
                print(json.dumps({"status":"already_completed","run":str(run_dir),**completed}),flush=True)
                return completed
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Run is nonempty; use --resume or choose a new --out")
    disk_guard(cfg)
    acc = accelerator_for(device)
    if resume and not (checkpoint_path(run_dir)/"training.pt").exists():
        raise ValueError("This run is weights-only; use --initialize into a new run, not --resume")
    model = RecurrentLM.build(cfg, acc.device,
        checkpoint_path(run_dir)/"backbone_config" if resume else None)
    tokenizer = tokenizer_for(cfg, checkpoint_path(run_dir) if resume else None)
    reference = reference_for(cfg, acc.device)
    optimizer = optimizer_for(model)
    progress = {"step":0,"episode":0,"processed_tokens":0,"scored_tokens":0}
    if initialize:
        source = Config.load(checkpoint_path(initialize)/"config.json")
        for key in ["model_id","revision","method","slots","tiny","tiny_arch"]:
            if key != "revision" and getattr(source,key)!=getattr(cfg,key):
                raise ValueError(f"Initialize mismatch: {key}")
        if not cfg.tiny and source.revision != cfg.revision:
            raise ValueError("Base revision differs from initialized checkpoint")
        load_weights(model, initialize)
    if resume:
        path = load_weights(model, run_dir)
        training = torch.load(path/"training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(training["optimizer"])
        progress = training["progress"]
        torch.set_rng_state(training["torch_rng"])
        if acc.device.type=="cuda":
            torch.cuda.set_rng_state_all(training["cuda_rng"])
        random.setstate(training["python_rng"])
        nr = training["numpy_rng"]
        np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]))
    model, optimizer = acc.prepare(model, optimizer)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir/"config.json")
    (run_dir/"environment.json").write_text(json.dumps({"versions":versions(),"device":str(acc.device),
        "gpu":torch.cuda.get_device_name() if acc.device.type=="cuda" else None,
        "native_context":model.native_context,"trainable":inventory(model),
        "training_mode":"full", "detach_local":cfg.detach_local,
        "kl_weight":cfg.kl_weight, "kl_direction":"reference || student"},indent=2)+"\n")
    documents = TokenDocuments(cfg.data_dir,"train") if cfg.natural_fraction else None
    if documents is not None: documents.verify_tokenizer(cfg)
    start_time, invocation_tokens = time.monotonic(), 0
    updates_here = 0
    model.train()
    if acc.device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    while progress["processed_tokens"] < cfg.token_budget:
        disk_guard(cfg)
        # Make accumulation weights explicit: equal episode mean losses, averaged per update.
        episodes, tokens = [], 0
        while tokens < cfg.tokens_per_update and progress["processed_tokens"]+tokens < cfg.token_budget:
            ep = training_episode(cfg,tokenizer,progress["episode"]+len(episodes),documents)
            episodes.append(ep); tokens += len(ep.prompt)+len(ep.answer)-1
        optimizer.zero_grad(set_to_none=True)
        losses=[]; kls=[]; scored=0
        for ep_index, ep in enumerate(episodes):
            ids, labels = ep.tensors(acc.device)
            with acc.autocast():
                loss, _ = model.episode_loss(ids,labels)
                weighted = loss
                if reference is not None and (progress["episode"]+ep_index) % cfg.kl_every == 0:
                    kl = retention_kl(model, reference, ids)
                    weighted = weighted + cfg.kl_weight * kl
                    kls.append(float(kl.detach()))
            if not torch.isfinite(weighted):
                raise FloatingPointError("Nonfinite loss; no checkpoint was overwritten")
            acc.backward(weighted / len(episodes))
            losses.append(float(loss.detach())); scored+=int((labels!=-100).sum())
        grad = acc.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient")
        fraction = progress["processed_tokens"] / cfg.token_budget
        ramp = min(1.0, (progress["processed_tokens"]+tokens)/(cfg.token_budget*cfg.warmup_fraction))
        scale = ramp * (0.1 + 0.9*0.5*(1+math.cos(math.pi*fraction)))
        for group in optimizer.param_groups: group["lr"]=group["base_lr"]*scale
        optimizer.step()
        progress["step"]+=1; progress["episode"]+=len(episodes)
        progress["processed_tokens"]+=tokens; progress["scored_tokens"]+=scored
        invocation_tokens+=tokens; updates_here+=1
        elapsed=time.monotonic()-start_time
        tps=invocation_tokens/max(elapsed,1e-8)
        row={**progress,"mean_episode_loss":sum(losses)/len(losses),"gradient_norm":float(grad),
            "mean_retention_kl":sum(kls)/len(kls) if kls else None,"kl_samples":len(kls),
            "tokens_per_second":tps,"eta_seconds":max(0,cfg.token_budget-progress["processed_tokens"])/tps,
            "peak_allocated_gb":torch.cuda.max_memory_allocated()/1e9 if acc.device.type=="cuda" else 0,
            "peak_reserved_gb":torch.cuda.max_memory_reserved()/1e9 if acc.device.type=="cuda" else 0}
        append_json(run_dir/"train.jsonl",row); print(json.dumps(row),flush=True)
        done = progress["processed_tokens"]>=cfg.token_budget or (max_updates and updates_here>=max_updates)
        if progress["step"]%cfg.save_every==0 or done:
            save_checkpoint(acc.unwrap_model(model),optimizer,run_dir,progress)
        if done: break
    if progress["processed_tokens"] >= cfg.token_budget:
        (run_dir/"TRAINING_COMPLETE.json").write_text(json.dumps(progress,indent=2)+"\n")
    return progress


def profile(cfg, length, steps, output, device=None):
    if steps < 2: raise ValueError("Profile at least two steps to include allocated optimizer states")
    if cfg.method == "native": raise ValueError("Profile a training method, not native inference")
    cfg.lengths=(length,);cfg.length_weights=(1.0,)
    acc=accelerator_for(device)
    model=RecurrentLM.build(cfg,acc.device)
    tokenizer=tokenizer_for(cfg)
    reference=reference_for(cfg,acc.device)
    optimizer=optimizer_for(model)
    model,optimizer=acc.prepare(model,optimizer)
    model.train()
    timings=[]
    if acc.device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        ids,labels=training_episode(cfg,tokenizer,i).tensors(acc.device)
        optimizer.zero_grad(set_to_none=True)
        start=time.monotonic()
        with acc.autocast():
            loss,state=model.episode_loss(ids,labels)
            if reference is not None: loss=loss+cfg.kl_weight*retention_kl(model,reference,ids)
        acc.backward(loss);acc.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if acc.device.type=="cuda": torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
    peak=torch.cuda.max_memory_reserved() if acc.device.type=="cuda" else 0
    total=torch.cuda.get_device_properties(acc.device).total_memory if acc.device.type=="cuda" else 0
    free_now=torch.cuda.mem_get_info(acc.device)[0] if total else 0
    result={"method":cfg.method,"length":length,"steps":steps,"seconds":timings,
        "processed_tokens_per_second":(length-1)/timings[-1],"last_loss":float(loss.detach()),
        "state_bytes":state.nbytes(),"peak_reserved_bytes":peak,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated() if total else 0,
        "gpu_total_bytes":total,"gpu_free_bytes_after_step":free_now,
        "headroom_ok":bool(total and min(total-peak,free_now)>=2*1024**3),
        "versions":versions(),"config":asdict(cfg),"training_mode":"full",
        "backbone_trainable_parameters":sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        "optimizer_state_bytes":sum(v.numel()*v.element_size() for st in optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor)),
        "backbone_dtypes":sorted({str(p.dtype) for p in model.backbone.parameters()}),
        "reference_loaded":reference is not None}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if total and not result["headroom_ok"]:
        raise RuntimeError("Less than 2 GiB GPU headroom. Reduce sequence length and re-profile")
    return result

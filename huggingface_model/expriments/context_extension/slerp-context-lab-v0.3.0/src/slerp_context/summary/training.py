"""Full-weight summary SFT with shared initialization and exact resume."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import math
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from ..data import tokenizer_for,Episode
from ..storage import (save_checkpoint,load_weights,checkpoint_path,append_json,
                       disk_guard,inventory,versions,tensor_bytes)
from ..train import optimizer_for
from .data import load_split,summarization_episode,prompt_parts,prompt_ids,encode
from .models import device_for,resolve_cfg,construct,restore,weight_identity
from .inference import amp,sync
from .config import digest,file_digest


def memory_distillation(model,teacher,episode,study,device):
    """Teacher sees all source; student uses its trained memory architecture."""
    n=min(study.distill_tokens,len(episode.answer))
    x=torch.tensor([episode.prompt+episode.answer[:n-1]],device=device)
    if x.shape[1]>model.native_context:return None,0
    with torch.no_grad():
        hidden=teacher.model(x,use_cache=False).last_hidden_state[:,-n:]
        reference=teacher.get_output_embeddings()(hidden).float().log_softmax(-1)
    student=model.tail_logits(x,n).float().log_softmax(-1)
    return F.kl_div(student.reshape(-1,student.shape[-1]),reference.reshape(-1,reference.shape[-1]),
                    log_target=True,reduction="batchmean"),x.shape[1]


def eligible_rows(rows,tokenizer,study,native):
    overhead=sum(map(len,prompt_parts(tokenizer,study)))
    cfg=study.base();kept=[];skipped={"source_limit":0,"target_limit":0,"native_limit":0}
    for row in rows:
        if row["source_tokens"]>study.train_source_limit:skipped["source_limit"]+=1;continue
        if row["reference_tokens"]>study.train_target_limit:skipped["target_limit"]+=1;continue
        if cfg.method=="native" and overhead+row["source_tokens"]+row["reference_tokens"]+1>native:
            skipped["native_limit"]+=1;continue
        kept.append(row)
    if not kept:raise ValueError(f"No whole source/target training pairs fit: {skipped}")
    return kept,skipped


def train(study,run_dir,device=None,initialize=None,resume=False,max_updates=None):
    run=Path(run_dir);device=device_for(device)
    if resume and initialize:raise ValueError("Choose resume or shared initialization")
    rows,meta=load_split(study,"train")
    cfg=resolve_cfg(study,meta);study.model=asdict(cfg)
    request=digest({"study":asdict(study),"data_sha256":meta["sha256"]})
    disk_guard(cfg)
    if resume:
        record=json.loads((run/"summary-run.json").read_text())
        if record["request_sha256"]!=request:raise ValueError("Resume study/data differ; start a new run")
        if (run/"TRAINING_COMPLETE.json").exists():return json.loads((run/"TRAINING_COMPLETE.json").read_text())
        model,path=restore(run,device)
        if not (path/"training.pt").exists():raise ValueError("Optimizer was finalized; cannot resume")
    else:
        if run.exists() and any(run.iterdir()):raise FileExistsError("Nonempty run; use --resume")
        model=construct(cfg,device,initialize);path=None
    tokenizer=tokenizer_for(cfg,path)
    rows,skipped=eligible_rows(rows,tokenizer,study,model.native_context)
    reference_init=(record["initialization"] if record["initialization"]!="HF_pretrained" else None) if resume else initialize
    if resume and reference_init and file_digest(checkpoint_path(reference_init)/"model.safetensors")!=record["initialization_sha256"]:
        raise ValueError("Frozen teacher/shared initialization changed since the run began")
    teacher=construct(cfg,device,reference_init).backbone.requires_grad_(False).eval() if study.distill_weight else None
    if teacher is not None and device.type=="cuda":teacher.to(dtype=torch.bfloat16)
    optimizer=optimizer_for(model)
    progress={"step":0,"epoch":0,"offset":0,"processed_tokens":0,"source_tokens":0,
              "scored_tokens":0,"documents":0,"teacher_tokens":0,"student_distill_tokens":0}
    if resume:
        saved=torch.load(path/"training.pt",map_location="cpu",weights_only=True)
        optimizer.load_state_dict(saved["optimizer"]);progress=saved["progress"]
        torch.set_rng_state(saved["torch_rng"]);random.setstate(saved["python_rng"])
        nr=saved["numpy_rng"];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),*nr[2:]))
        if device.type=="cuda":torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        run.mkdir(parents=True,exist_ok=True)
        cfg.save(run/"config.json");study.save(run/"study.json")
        record={"request_sha256":request,"data_manifest":meta,"eligible_documents":len(rows),
            "skipped_documents":skipped,"initialization":str(initialize) if initialize else "HF_pretrained",
            "initialization_sha256":file_digest(checkpoint_path(initialize)/"model.safetensors") if initialize else None,
            "training_mode":"full","loss":"mean_document_summary_CE",
            "memory_distillation_direction":"full_source_teacher || compressed_student",
            "teacher_policy":"frozen_initial_backbone; only when complete teacher input fits native limit",
            "versions":versions(),"native_context":model.native_context,"trainable":inventory(model)}
        (run/"summary-run.json").write_text(json.dumps(record,indent=2)+"\n")
    model.train();start=time.perf_counter();initial_tokens=progress["processed_tokens"];updates=0
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    while progress["processed_tokens"]<cfg.token_budget and progress["epoch"]<study.max_epochs:
        disk_guard(cfg);batch=[];logical=0
        while logical<cfg.tokens_per_update and progress["epoch"]<study.max_epochs:
            order=list(range(len(rows)));random.Random(cfg.seed+progress["epoch"]).shuffle(order)
            row=rows[order[progress["offset"]]]
            ep=summarization_episode(row,tokenizer,study)
            n=len(ep.prompt)+len(ep.answer)-1;batch.append((row,ep));logical+=n
            progress["offset"]+=1
            if progress["offset"]==len(rows):progress["epoch"]+=1;progress["offset"]=0
            if progress["processed_tokens"]+logical>=cfg.token_budget:break
        optimizer.zero_grad(set_to_none=True);losses=[];kls=[];kd_skipped=0;state_bytes=0
        for row,ep in batch:
            ids,labels=ep.tensors(device)
            with amp(device):
                loss,state=model.episode_loss(ids,labels);weighted=loss
                if teacher is not None and progress["documents"]%study.distill_every==0:
                    kd,n=memory_distillation(model,teacher,ep,study,device)
                    if kd is None:kd_skipped+=1
                    else:
                        weighted=weighted+study.distill_weight*kd;kls.append(float(kd.detach()))
                        progress["teacher_tokens"]+=n;progress["student_distill_tokens"]+=n
            if not torch.isfinite(weighted):raise FloatingPointError("Nonfinite training objective")
            (weighted/len(batch)).backward()
            losses.append(float(loss.detach()));state_bytes=max(state_bytes,state.nbytes())
            progress["scored_tokens"]+=int((labels!=-100).sum());progress["documents"]+=1
            progress["source_tokens"]+=row["source_tokens"]
            del state,weighted,loss
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad):raise FloatingPointError("Nonfinite gradient")
        fraction=min(1.0,progress["processed_tokens"]/cfg.token_budget)
        warm=min(1.0,(progress["processed_tokens"]+logical)/(cfg.token_budget*cfg.warmup_fraction))
        scale=warm*(0.1+0.9*0.5*(1+math.cos(math.pi*fraction)))
        for group in optimizer.param_groups:group["lr"]=group["base_lr"]*scale
        optimizer.step();sync(device)
        progress["step"]+=1;progress["processed_tokens"]+=logical;updates+=1
        elapsed=time.perf_counter()-start;speed=(progress["processed_tokens"]-initial_tokens)/max(elapsed,1e-9)
        rowlog={**progress,"mean_summary_loss":sum(losses)/len(losses),"mean_memory_kl":sum(kls)/len(kls) if kls else None,
            "distill_skipped_native_limit":kd_skipped,"logical_tokens_per_second":speed,
            "eta_to_token_budget_s":max(0,cfg.token_budget-progress["processed_tokens"])/max(speed,1e-9),
            "state_bytes":state_bytes,"gradient_norm":float(grad),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "document_ids":[r["id"] for r,e in batch]}
        append_json(run/"summary-train.jsonl",rowlog);print(json.dumps(rowlog),flush=True)
        complete=progress["processed_tokens"]>=cfg.token_budget or progress["epoch"]>=study.max_epochs
        interrupted=max_updates is not None and updates>=max_updates
        if complete or interrupted or progress["step"]%cfg.save_every==0:save_checkpoint(model,optimizer,run,progress)
        if complete:
            result={**progress,"stop_reason":"token_budget" if progress["processed_tokens"]>=cfg.token_budget else "max_epochs",
                    "complete":True,"full_weight_training":True}
            (run/"TRAINING_COMPLETE.json").write_text(json.dumps(result,indent=2)+"\n")
            return result
        if interrupted:return {**progress,"complete":False}
    return progress


def profile(study,source_tokens,steps,output,device=None,initialize=None):
    if steps<2:raise ValueError("At least two updates are needed to allocate optimizer state")
    if Path(output).exists():raise FileExistsError("Choose a fresh profile output")
    from huggingface_hub import HfApi
    cfg=study.base();device=device_for(device)
    if not cfg.tiny:cfg.revision=HfApi().model_info(cfg.model_id,revision=cfg.revision).sha
    model=construct(cfg,device,initialize);tokenizer=tokenizer_for(cfg)
    unit=encode(tokenizer,"The study assessed alternative proposals and identified qualified benefits. ")
    source=(unit*((source_tokens+len(unit)-1)//len(unit)))[:source_tokens]
    target=encode(tokenizer,"The report evaluates the evidence and recommends further assessment. ")
    target=(target*((study.train_target_limit+len(target)-1)//len(target)))[:study.train_target_limit]
    ep=Episode(prompt_ids(tokenizer,study,source),target,"",{"task":"summary_profile"})
    if cfg.method=="native" and len(ep.prompt)+len(ep.answer)>model.native_context:
        raise ValueError("Native profile exceeds context; choose a shorter source shape")
    model.train();optimizer=optimizer_for(model);times=[]
    teacher=copy.deepcopy(model.backbone).requires_grad_(False).eval() if study.distill_weight else None
    if teacher is not None and device.type=="cuda":teacher.to(dtype=torch.bfloat16)
    ids,labels=ep.tensors(device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    result={"method":cfg.method,"source_tokens":source_tokens,"target_tokens":len(target),
            "steps":steps,"profile_only_synthetic_shape":True,"quality_claim":False,
            "study_sha256":digest(asdict(study)),"model_revision":cfg.revision,
            "device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
            "versions":versions(),"full_weight_training":True}
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True);sync(device);start=time.perf_counter()
            with amp(device):
                loss,state=model.episode_loss(ids,labels)
                if teacher is not None:
                    kd,_=memory_distillation(model,teacher,ep,study,device)
                    if kd is not None:loss=loss+study.distill_weight*kd
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();sync(device)
            times.append(time.perf_counter()-start)
        result.update(status="ok",seconds_per_update=times,final_loss=float(loss.detach()),state_bytes=state.nbytes(),
                      optimizer_bytes=tensor_bytes(optimizer.state_dict()),
                      trainable_backbone_parameters=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad))
    except torch.OutOfMemoryError:
        result.update(status="oom",seconds_per_update=times)
    if device.type=="cuda":
        total=torch.cuda.get_device_properties(device).total_memory
        peak=torch.cuda.max_memory_reserved(device);free=torch.cuda.mem_get_info(device)[0]
        result.update(peak_reserved_bytes=peak,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      gpu_total_bytes=total,free_bytes_after=free,
                      headroom_ok=result["status"]=="ok" and min(total-peak,free)>=2*1024**3)
    else:result.update(headroom_ok=None,cuda_profile=False)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    if result["status"]!="ok" or result["headroom_ok"] is False:
        raise RuntimeError("GPU profile failed or has less than 2 GiB headroom; lower source length and repeat")
    return result

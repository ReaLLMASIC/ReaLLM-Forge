"""Optional single-update, group-relative policy optimization with prefix replay.

Explicit loop instead of an unverified TRL cache override. Full-softmax sampling,
temperature 1, four completions by default. Reference has its own memory weights.
"""
from pathlib import Path
import json
import time
import torch
from .data import tokenizer_for, synthetic
from .evaluate import load_run, exact
from .storage import append_json, save_checkpoint, disk_guard
from .train import accelerator_for, optimizer_for


def grpo_loss(logp, old_logp, ref_logp, advantage, beta=0.02, clip=0.2):
    ratio=(logp-old_logp).exp()
    unclipped=ratio*advantage
    clipped=ratio.clamp(1-clip,1+clip)*advantage
    delta=ref_logp-logp
    kl=delta.exp()-delta-1
    return (-torch.minimum(unclipped,clipped)+beta*kl).mean(),kl.mean()


def grpo(checkpoint, output, groups=200, group_size=4, length=4096,
         learning_rate=1e-5, beta=0.02, device=None):
    if group_size<2: raise ValueError("At least two completions per group")
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Use a new RL output directory")
    acc=accelerator_for(device)
    policy,tok=load_run(checkpoint,str(acc.device))
    reference,_=load_run(checkpoint,str(acc.device));reference.requires_grad_(False);reference.eval()
    policy.cfg.memory_lr=learning_rate;policy.cfg.backbone_lr=learning_rate
    # SFT checkpoint retains maximum adaptation length; record RL limit separately.
    if length>max(policy.cfg.lengths): raise ValueError("RL length exceeds declared training limit")
    optimizer=optimizer_for(policy)
    policy,optimizer=acc.prepare(policy,optimizer)
    out.mkdir(parents=True,exist_ok=True)
    (out/"rl_config.json").write_text(json.dumps({"source":str(checkpoint),"groups":groups,
        "group_size":group_size,"length":length,"lr":learning_rate,"beta":beta,
        "temperature":1.0,"top_k":0,"replay_prefix":True,"seed":42873},indent=2)+"\n")
    torch.manual_seed(42873)
    progress={"step":0,"episode":0,"processed_tokens":0,"scored_tokens":0,"optimizer_updates":0}
    varying=0
    for group in range(groups):
        disk_guard(policy.cfg)
        ep=synthetic(tok,length,42873,group,task="recall",fact_count=2,
            window=policy.cfg.window,chunk=policy.cfg.chunk,chat_template=policy.cfg.chat_template)
        prompt=torch.tensor([ep.prompt],device=acc.device)
        completions=[];old=[];refs=[];rewards=[]
        # Each sibling reconstructs its own prompt state. No mutable cache is shared.
        with torch.no_grad(),acc.autocast():
            for _ in range(group_size):
                completion,_=policy.generate_stream(prompt,min(64,len(ep.answer)+8),temperature=1,
                    top_k=0,eos_token_id=tok.eos_token_id)
                completions.append(completion)
                old.append(policy.completion_logps(prompt,completion).detach())
                refs.append(reference.completion_logps(prompt,completion).detach())
                rewards.append(exact(tok.decode(completion[0],skip_special_tokens=True),ep.target))
        r=torch.tensor(rewards,device=acc.device)
        std=r.std(unbiased=False)
        advantage=(r-r.mean())/(std+1e-6)
        optimizer.zero_grad(set_to_none=True)
        policy.train()
        loss_values=[];kl_values=[]
        if float(std)>0:
            varying+=1
            for i,completion in enumerate(completions):
                with acc.autocast():
                    current=policy.completion_logps(prompt,completion)
                    loss,kl=grpo_loss(current,old[i],refs[i],advantage[i],beta)
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite GRPO objective")
                acc.backward(loss/group_size)
                loss_values.append(float(loss.detach()));kl_values.append(float(kl.detach()))
            grad=acc.clip_grad_norm_(policy.parameters(),1.0)
            if not torch.isfinite(grad): raise FloatingPointError("Nonfinite RL gradient")
            optimizer.step();progress["optimizer_updates"]+=1
        # Count actual token forward work: rollout + old replay + reference replay [+ current replay].
        work=sum((len(ep.prompt)+c.numel())*(4 if float(std)>0 else 3) for c in completions)
        progress["step"]+=1;progress["episode"]+=group_size
        progress["processed_tokens"]+=work;progress["scored_tokens"]+=sum(c.numel() for c in completions)
        row={**progress,"rewards":rewards,"varying_reward_fraction":varying/(group+1),
            "loss":sum(loss_values)/len(loss_values) if loss_values else None,
            "kl":sum(kl_values)/len(kl_values) if kl_values else None}
        append_json(out/"rl.jsonl",row);print(json.dumps(row),flush=True)
        if (group+1)%25==0 or group+1==groups:
            save_checkpoint(acc.unwrap_model(policy),optimizer,out,progress)
    return progress

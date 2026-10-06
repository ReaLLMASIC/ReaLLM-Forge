from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import torch
from .config import Config


def parser():
    p=argparse.ArgumentParser(description="SLERP context lab v0.2.0 — full-weight training")
    commands=p.add_subparsers(dest="command",required=True)
    def config_arg(s):
        s.add_argument("--config",default="configs/hunyuan-slerp.json")
        s.add_argument("--device",choices=["cpu","cuda"])
    d=commands.add_parser("doctor");config_arg(d)
    d=commands.add_parser("download");config_arg(d)
    t=commands.add_parser("train");config_arg(t)
    t.add_argument("--out",required=True);t.add_argument("--resume",action="store_true")
    t.add_argument("--initialize");t.add_argument("--max-updates",type=int)
    t.add_argument("--token-budget",type=int)
    prof=commands.add_parser("profile");config_arg(prof)
    prof.add_argument("--length",type=int,default=8192);prof.add_argument("--steps",type=int,default=3)
    prof.add_argument("--out",default="reports/profile.json")
    data=commands.add_parser("prepare-data");config_arg(data)
    data.add_argument("--out",default="data/pg19");data.add_argument("--split",choices=["train","validation","test"],default="train")
    data.add_argument("--tokens",type=int,default=25_000_000);data.add_argument("--revision",default="main")
    ev=commands.add_parser("evaluate")
    source=ev.add_mutually_exclusive_group(required=True)
    source.add_argument("--run");source.add_argument("--config")
    ev.add_argument("--out",required=True)
    ev.add_argument("--lengths",nargs="+",type=int,default=[4096,8192,16384,32768])
    ev.add_argument("--samples",type=int,default=4)
    ev.add_argument("--tasks",nargs="+",default=["recall","update","trace","multi"])
    ev.add_argument("--facts",nargs="+",type=int,default=[1,8,32])
    ev.add_argument("--positions",nargs="+",type=float,default=[.05,.5,.9])
    ev.add_argument("--seed",type=int,default=91991);ev.add_argument("--device",choices=["cpu","cuda"])
    ev.add_argument("--disable-memory",action="store_true")
    ev.add_argument("--max-new-tokens",type=int,default=128)
    assess=commands.add_parser("assess")
    assess.add_argument("inputs",nargs="+");assess.add_argument("--out",required=True)
    assess.add_argument("--threshold",type=float)
    assess.add_argument("--metric",choices=["exact_match","value_exact_match"],default="value_exact_match")
    final=commands.add_parser("finalize",help="Remove optimizer state explicitly; preserve complete weights")
    final.add_argument("--run",required=True)
    parity=commands.add_parser("parity");config_arg(parity)
    parity.add_argument("--out",default="reports/native-parity.json")
    parity.add_argument("--length",type=int,default=512)
    ppl=commands.add_parser("perplexity")
    ppl.add_argument("--run",required=True);ppl.add_argument("--data-dir",default="data/pg19")
    ppl.add_argument("--out",required=True);ppl.add_argument("--lengths",nargs="+",type=int,default=[2048,4096,8192,16384,32768])
    ppl.add_argument("--samples",type=int,default=8);ppl.add_argument("--suffix",type=int,default=512)
    ppl.add_argument("--split",choices=["validation","test"],default="validation");ppl.add_argument("--device",choices=["cpu","cuda"])
    gen=commands.add_parser("generate")
    gen.add_argument("--run",required=True);gen.add_argument("--prompt",required=True)
    gen.add_argument("--tokens",type=int,default=128);gen.add_argument("--temperature",type=float,default=0)
    gen.add_argument("--top-k",type=int,default=0);gen.add_argument("--device",choices=["cpu","cuda"])
    hist=commands.add_parser("generated-history")
    hist.add_argument("--run",required=True);hist.add_argument("--out",required=True)
    hist.add_argument("--samples",type=int,default=8);hist.add_argument("--gap",type=int,default=2304)
    hist.add_argument("--device",choices=["cpu","cuda"])
    pred=commands.add_parser("predict-jsonl")
    source=pred.add_mutually_exclusive_group(required=True)
    source.add_argument("--run");source.add_argument("--config")
    pred.add_argument("--input",required=True);pred.add_argument("--out",required=True)
    pred.add_argument("--device",choices=["cpu","cuda"])
    pred.add_argument("--chat-template",action="store_true")
    pred.add_argument("--max-new-tokens",type=int,default=128)
    rl=commands.add_parser("grpo")
    rl.add_argument("--run",required=True);rl.add_argument("--out",required=True)
    rl.add_argument("--groups",type=int,default=200);rl.add_argument("--group-size",type=int,default=4)
    rl.add_argument("--length",type=int,default=4096);rl.add_argument("--lr",type=float,default=1e-5)
    rl.add_argument("--beta",type=float,default=.02);rl.add_argument("--device",choices=["cpu","cuda"])
    plots=commands.add_parser("plot");plots.add_argument("inputs",nargs="+");plots.add_argument("--out",default="reports/figures")
    plots.add_argument("--metric",choices=["exact_match","value_exact_match"],default="value_exact_match")
    arc=commands.add_parser("archive");arc.add_argument("--run",required=True);arc.add_argument("--destination",required=True)
    arc.add_argument("--mbps",type=float,default=20)
    return p


def main():
    args=parser().parse_args()
    if not torch.cuda.is_available(): torch.set_num_threads(min(4,os.cpu_count() or 1))
    if getattr(args,"config",None): cfg=Config.load(args.config)
    if args.command=="doctor":
        from .storage import disk_guard,versions
        print(json.dumps({"versions":versions(),"cuda_available":torch.cuda.is_available(),
            "gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "bf16":torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "disk":disk_guard(cfg)},indent=2))
        if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    elif args.command=="download":
        from huggingface_hub import HfApi,snapshot_download
        from .storage import disk_guard
        info=HfApi().model_info(cfg.model_id,revision=cfg.revision,files_metadata=True)
        sha=info.sha
        estimate=sum((f.size or 0) for f in info.siblings if f.rfilename.endswith('.safetensors'))+100_000_000
        disk_guard(cfg,estimate)
        snapshot_download(cfg.model_id,revision=sha,allow_patterns=["config.json","*.safetensors",
            "*.safetensors.index.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json",
            "generation_config.json","vocab.json","merges.txt","*.model","added_tokens.json"])
        from .config import revision_lock
        p=revision_lock(cfg.model_id);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"model_id":cfg.model_id,"revision":sha},indent=2)+"\n")
        print(p.read_text())
    elif args.command=="train":
        from .train import train
        if args.token_budget: cfg.token_budget=args.token_budget
        train(cfg,args.out,args.resume,args.initialize,args.device,args.max_updates)
    elif args.command=="profile":
        from .train import profile
        try:
            profile(cfg,args.length,args.steps,args.out,args.device)
        except torch.cuda.OutOfMemoryError:
            from dataclasses import asdict
            result={"status":"out_of_memory","length":args.length,"config":asdict(cfg),
                "headroom_ok":False,"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
            out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(result,indent=2)+"\n")
            raise
    elif args.command=="prepare-data":
        from .data import prepare_documents
        prepare_documents(cfg,args.out,args.tokens,args.split,args.revision)
    elif args.command=="evaluate":
        from .evaluate import evaluate
        evaluate(args.run,args.out,args.lengths,args.samples,args.tasks,args.facts,args.positions,args.seed,args.device,args.disable_memory,args.config,args.max_new_tokens)
    elif args.command=="assess":
        from .evaluate import assess
        assess(args.inputs,args.out,args.threshold,args.metric)
    elif args.command=="finalize":
        from .storage import finalize_run
        print(json.dumps(finalize_run(args.run),indent=2))
    elif args.command=="parity":
        from .validation import native_parity
        native_parity(cfg,args.out,args.device,args.length)
    elif args.command=="perplexity":
        from .evaluate import perplexity
        perplexity(args.run,args.data_dir,args.out,args.lengths,args.suffix,args.samples,args.split,args.device)
    elif args.command=="generated-history":
        from .evaluate import generated_history
        generated_history(args.run,args.out,args.samples,args.gap,args.device)
    elif args.command=="predict-jsonl":
        from .evaluate import predict_jsonl
        predict_jsonl(args.run,args.input,args.out,args.device,args.config,args.chat_template,args.max_new_tokens)
    elif args.command=="generate":
        from .evaluate import load_run,autocast,stop_ids
        model,tok=load_run(args.run,args.device)
        inp=torch.tensor([tok.encode(args.prompt,add_special_tokens=False)],device=next(model.parameters()).device)
        with autocast(model):
            output,state=model.generate_stream(inp,args.tokens,args.temperature,args.top_k,eos_token_id=stop_ids(model,tok))
        print(tok.decode(output[0],skip_special_tokens=True))
    elif args.command=="grpo":
        from .rl import grpo
        grpo(args.run,args.out,args.groups,args.group_size,args.length,args.lr,args.beta,args.device)
    elif args.command=="plot":
        from .plots import plots
        print(plots(args.inputs,args.out,args.metric))
    elif args.command=="archive":
        from .storage import archive_run
        print(archive_run(args.run,args.destination,args.mbps))


if __name__=="__main__":
    main()

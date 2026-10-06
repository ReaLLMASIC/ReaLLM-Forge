"""Hugging Face long-document summarization comparison."""
import argparse
import json
from .config import Study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    def study_command(name):
        q=sub.add_parser(name);q.add_argument("--config",default="configs/summary-smol.json")
        return q
    q=study_command("prepare")
    q.add_argument("--splits",nargs="+",choices=["train","validation","test"],default=["train","validation","test"])
    q=study_command("profile")
    q.add_argument("--method");q.add_argument("--source-tokens",type=int,default=16384)
    q.add_argument("--steps",type=int,default=2);q.add_argument("--out",required=True)
    q.add_argument("--device");q.add_argument("--initialize")
    q=study_command("train")
    q.add_argument("--method");q.add_argument("--out",required=True);q.add_argument("--device")
    q.add_argument("--initialize");q.add_argument("--resume",action="store_true")
    q.add_argument("--max-updates",type=int)
    q=study_command("evaluate")
    from .inference import BASELINES,LEARNED
    q.add_argument("--method",choices=sorted(BASELINES|LEARNED),required=True)
    q.add_argument("--out",required=True);q.add_argument("--checkpoint");q.add_argument("--device")
    q.add_argument("--split",choices=["validation","test"],default="validation")
    q.add_argument("--limit",type=int);q.add_argument("--min-source-tokens",type=int,default=0)
    q.add_argument("--max-source-tokens",type=int)
    q=sub.add_parser("report");q.add_argument("files",nargs="+");q.add_argument("--out",required=True)
    q.add_argument("--baseline",default="rolling");q.add_argument("--semantic-file")
    q=sub.add_parser("semantic");q.add_argument("files",nargs="+");q.add_argument("--out",required=True)
    q.add_argument("--encoder",default="sentence-transformers/all-MiniLM-L6-v2")
    q.add_argument("--revision",default="main");q.add_argument("--device",default="cpu")
    q.add_argument("--block-tokens",type=int,default=192)
    q=sub.add_parser("finalize");q.add_argument("run")
    a=p.parse_args();study=Study.load(a.config) if hasattr(a,"config") else None
    if study is not None and getattr(a,"method",None) and a.command in {"train","profile"}:
        study.model["method"]=a.method;study.validate()
    if a.command=="prepare":
        from .data import prepare_hf
        result=prepare_hf(study,a.splits)
    elif a.command=="profile":
        from .training import profile
        result=profile(study,a.source_tokens,a.steps,a.out,a.device,a.initialize)
    elif a.command=="train":
        from .training import train
        if a.max_updates is not None and a.max_updates<1:p.error("max-updates must be positive")
        result=train(study,a.out,a.device,a.initialize,a.resume,a.max_updates)
    elif a.command=="evaluate":
        from .evaluation import evaluate
        if a.limit is not None and a.limit<1:p.error("limit must be positive")
        result=evaluate(study,a.method,a.out,a.split,a.checkpoint,a.device,a.limit,a.min_source_tokens,a.max_source_tokens)
    elif a.command=="report":
        from .report import report
        result=report(a.files,a.out,a.baseline,a.semantic_file)
    elif a.command=="semantic":
        from .semantic import semantic
        result=semantic(a.files,a.out,a.encoder,a.revision,a.device,a.block_tokens)
    elif a.command=="finalize":
        from ..storage import finalize_run
        result=finalize_run(a.run)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

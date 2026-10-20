"""Run with python -m lane_lab; no installation or global entry point required."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Per-column transformer forecasting lab")
    sub=p.add_subparsers(dest="command", required=True)
    for name in ["train", "sweep"]:
        q=sub.add_parser(name)
        q.add_argument("--config")
        q.add_argument("--init-model", help="Fine-tune a checkpoint with fresh optimizer; uses its model config")
        q.add_argument("--channels", type=int, nargs="+" if name=="sweep" else None, default=[1,2,3] if name=="sweep" else 1)
        q.add_argument("--variant",choices=["independent","slerp"],default="slerp")
        q.add_argument("--knots",type=int,default=5)
        q.add_argument("--tied",action=argparse.BooleanOptionalAction,default=True)
        q.add_argument("--backend",choices=["auto","torch","triton"],default="auto")
        q.add_argument("--levels",type=int,default=128)
        q.add_argument("--width",type=int,default=48)
        q.add_argument("--layers",type=int,default=2)
        q.add_argument("--context",type=int,default=64)
        q.add_argument("--steps",type=int,default=1500)
        q.add_argument("--batch-size",type=int,default=32)
        q.add_argument("--lr",type=float,default=.003)
        q.add_argument("--seed",type=int,default=1337)
        q.add_argument("--threads",type=int,default=1)
        q.add_argument("--device",default="cpu")
        q.add_argument("--eval-every",type=int,default=250)
        q.add_argument("--horizon",type=int,default=32)
        q.add_argument("--mse-weight",type=float,default=10.)
        q.add_argument("--target-mode", choices=["hard", "linear"], default="hard",
                       help="Linear distributes numeric target mass between neighboring levels")
        q.add_argument("--input-noise",type=float,default=.004)
        q.add_argument("--rollout-steps",type=int,default=4)
        q.add_argument("--rollout-every",type=int,default=4)
        q.add_argument("--rollout-weight",type=float,default=10.)
        q.add_argument("--rollout-gradient", choices=["detached", "full"], default="detached")
        q.add_argument("--output",default="runs/sine" if name=="sweep" else "runs/c1")
    q=sub.add_parser("evaluate")
    q.add_argument("--model",required=True); q.add_argument("--output",required=True)
    q.add_argument("--device",default="cpu"); q.add_argument("--threads",type=int,default=1)
    q.add_argument("--horizon",type=int,default=32);q.add_argument("--sequences",type=int,default=24)
    q.add_argument("--seed",type=int,default=30000)
    q.add_argument("--mode",choices=["mean","greedy","sample"],default="mean")
    q=sub.add_parser("serve")
    q.add_argument("--runs",default="examples/checkpoints");q.add_argument("--port",type=int,default=8765)
    q.add_argument("--host",default="127.0.0.1");q.add_argument("--device",default="cpu")
    q.add_argument("--threads",type=int,default=1)
    q=sub.add_parser("report")
    q.add_argument("--runs",default="runs/sine");q.add_argument("--output",default="runs/sine/report")
    args=p.parse_args()
    if args.command in ["train","sweep"]:
        from .experiment import train,report
        for key in ["steps","batch_size","eval_every","horizon","rollout_every","threads"]:
            if getattr(args,key)<1: p.error(f"{key} must be positive")
        for key in ["rollout_steps", "mse_weight", "rollout_weight", "input_noise"]:
            if getattr(args,key)<0: p.error(f"{key} must be nonnegative")
        if args.lr <= 0: p.error("lr must be positive")
        if args.command=="train": train(args)
        else:
            if args.config or args.init_model:
                p.error("--config and --init-model are for train; sweep constructs its own models")
            root=Path(args.output)
            for c in args.channels:
                for variant,knots,tied in [("independent",2,True),("slerp",2,True),
                    ("slerp",2,False),("slerp",3,True),("slerp",3,False),
                    ("slerp",5,True),("slerp",5,False),("slerp",9,True),("slerp",9,False)]:
                    run=argparse.Namespace(**vars(args))
                    run.channels=c;run.variant=variant;run.knots=knots;run.tied=tied
                    run.output=str(root/f"c{c}-{variant}{knots if variant=='slerp' else ''}-{'tied' if tied else 'untied'}")
                    if (Path(run.output)/"metrics.json").exists():
                        print(f"Skipping completed run {run.output}")
                        continue
                    train(run)
            print(report(root,root/"report"))
    elif args.command=="evaluate":
        import torch
        from .model import LaneModel
        from .experiment import evaluate_model
        torch.set_num_threads(args.threads)
        model=LaneModel.from_pretrained(args.model).to(args.device)
        metrics,example=evaluate_model(model,args.seed,args.sequences,args.horizon,args.mode)
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
        (out/"example.json").write_text(json.dumps(example))
        print(json.dumps(metrics,indent=2))
    elif args.command=="serve":
        from .server import serve
        serve(args.runs,host=args.host,port=args.port,device=args.device,threads=args.threads)
    else:
        from .experiment import report
        print(report(args.runs,args.output))


if __name__=="__main__": main()

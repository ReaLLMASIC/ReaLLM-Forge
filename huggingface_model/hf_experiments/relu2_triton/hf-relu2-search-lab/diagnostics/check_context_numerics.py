"""Print/save precision diagnostics for the reported cached-attention fixture."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import torch
from .context_precision import precision_oracle, assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=928)
    parser.add_argument('--alpha',type=float,choices=[0.,.5,1.],default=0.)
    parser.add_argument('--output',type=Path,default=Path('runs/context_numerics.json'))
    parser.add_argument('--cpu-emulation',action='store_true',help='Emulate cast locations only; NOT a CUDA test')
    a=parser.parse_args()
    if a.cpu_emulation:device='cpu'
    else:
        device=a.device
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            parser.error('CUDA is required unless --cpu-emulation is explicitly supplied')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    q=torch.randn(1,3,3,128,device=device,dtype=torch.bfloat16).div(8).requires_grad_()
    k=torch.randn(1,3,1057,128,device=device,dtype=torch.bfloat16).div(8).requires_grad_()
    v=torch.randn_like(k,requires_grad=True)
    scale=torch.tensor(-.7,device=device,requires_grad=True)
    bias=torch.tensor([-.2,.15,0.],device=device,requires_grad=True)
    if not a.cpu_emulation:
        from hf_model.triton_context_attention import triton_context_attention
        y=triton_context_attention(q,k,v,scale=scale,bias=bias,divisor=2.,alpha=a.alpha,anchor=8.)
    do=torch.randn_like(q)
    oracle=precision_oracle(q,k,v,do,scale,bias,alpha=a.alpha)
    if a.cpu_emulation:
        actual={name:p.rounded for name,p in oracle.items()}
    else:
        y.backward(do)
        actual=dict(output=y,dq=q.grad,dk=k.grad,dv=v.grad,dscale=scale.grad,dbias=bias.grad)
    reports={name:assess(name,value,oracle[name]) for name,value in actual.items()}
    payload=dict(mode='CPU cast emulation; no CUDA execution' if a.cpu_emulation else 'actual fused CUDA',
        seed=a.seed,alpha=a.alpha,shape=[1,3,3,1057,128],torch=torch.__version__,
        python=platform.python_version(),device=device,triton=importlib.metadata.version('triton'),
        gpu=torch.cuda.get_device_name(device) if not a.cpu_emulation else None,
        passed=all(r['passed'] for r in reports.values()),checks=reports,
        note='The old elementwise FP32-style tolerance is retained as a diagnostic. Acceptance requires every element within the arithmetic-derived rounding interval AND the unchanged 1.5% mathematical RMS limit. FP64 ideal math is independently tested against autograd.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not payload['passed'] and not a.cpu_emulation:
        dump=a.output.with_suffix('.tensors.pt')
        torch.save(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),do=do.detach().cpu(),
            scale=scale.detach().cpu(),bias=bias.detach().cpu(),alpha=a.alpha,
            actual={name:value.detach().cpu() for name,value in actual.items()}),dump)
        payload['failure_tensors']=str(dump)
    temp=a.output.with_suffix(a.output.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps(payload,indent=2,allow_nan=False))
    print(f'Saved {a.output}')
    if not payload['passed']:raise SystemExit(1)


if __name__=='__main__':main()

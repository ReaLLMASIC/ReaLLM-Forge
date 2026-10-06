"""Execute the actual Triton tiles in its CPU interpreter or compile for GPUs.

Interpreter: TRITON_INTERPRET=1 python scripts/check_context_kernels.py --interpreter
Compilation: python scripts/check_context_kernels.py --compile
Neither mode measures GPU execution or training quality.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton


def modules():
    # Explicit fixed tiles avoid an autotuning call needing a CUDA device.
    original=triton.autotune
    triton.autotune=lambda *a,**k:lambda f:f
    try:
        from hf_model import triton_context_attention as kernels
    finally:
        triton.autotune=original
    return kernels


def interpreted(kernels, m, n, alpha, scale_value, use_mask, use_bias, splits=1, offset=None, causal=True):
    from hf_model.context_attention import context_attention_reference
    b,h,d=2,3,16
    torch.manual_seed(97)
    q=torch.randn(b,h,m,d)/4
    k=torch.randn(b,h,n,d)/4
    v=torch.randn(b,h,n,d)
    scale=torch.tensor(scale_value,requires_grad=True)
    bias=torch.tensor([-.2,0.,.25],requires_grad=True) if use_bias else None
    mask=torch.ones(b,n,dtype=torch.bool) if use_mask else None
    if mask is not None:
        mask[0,:min(4,n)]=False
        mask[1,1::3]=False
    offset=n-m if offset is None else offset
    counts=mask.to(torch.int32).cumsum(-1,dtype=torch.int32).contiguous() if mask is not None else None
    parts=torch.empty(splits,b,h,m,d)
    opts=kernels._launch_options(q,k,v,q,scale,2.,causal,offset,mask,bias,alpha,3.5)
    tile=dict(BLOCK_M=16,BLOCK_N=16,num_warps=4,num_stages=1)
    kernels._rows_kernel[(triton.cdiv(m,16),b*h,splits)](q,k,v,parts,q,q,scale,
        mask if mask is not None else q,q,bias if bias is not None else q,
        counts if counts is not None else q,**opts,**tile,
        BACKWARD=False,NEED_DQ=False,NEED_DS=False,NEED_DB=False,SPLITS=splits)
    out=torch.empty_like(q)
    kernels._reduce_splits[(triton.cdiv(out.numel(),128),)](parts,out,SIZE=out.numel(),SPLITS=splits,
        BLOCK=128,BLOCK_SPLITS=triton.next_power_of_2(splits),num_warps=4)
    do=torch.randn_like(out)
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    ds,db=torch.empty(b,h,m),torch.empty(b,h,m)
    opts=kernels._launch_options(q,k,v,do,scale,2.,causal,offset,mask,bias,alpha,3.5)
    kernels._rows_kernel[(triton.cdiv(m,16),b*h,1)](q,k,v,dq,do,ds,scale,
        mask if mask is not None else q,db,bias if bias is not None else q,
        counts if counts is not None else q,**opts,**tile,
        BACKWARD=True,NEED_DQ=True,NEED_DS=True,NEED_DB=use_bias,SPLITS=1)
    kernels._kv_kernel[(triton.cdiv(n,16),b*h)](q,k,v,do,dk,dv,scale,
        mask if mask is not None else q,bias if bias is not None else q,
        counts if counts is not None else q,**opts,**tile,NEED_DK=True,NEED_DV=True)
    q.requires_grad_();k.requires_grad_();v.requires_grad_()
    reference=context_attention_reference(q,k,v,scale,bias,divisor=2.,alpha=alpha,anchor=3.5,
        attention_mask=mask,causal_offset=offset,causal=causal)
    reference.backward(do)
    for got,want in [(out,reference),(dq,q.grad),(dk,k.grad),(dv,v.grad),(ds.sum(),scale.grad)]+(
        [(db.sum((0,2)),bias.grad)] if use_bias else []):
        torch.testing.assert_close(got,want,atol=2e-5,rtol=2e-4)
    return dict(m=m,n=n,alpha=alpha,scale=scale_value,mask=use_mask,bias=use_bias,splits=splits,offset=offset,causal=causal,status="passed")


def compile_cases(kernels):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    results=[]
    # Production head size, regular/split-cache paths and every length exponent.
    for sm,alpha,use_bias,mode,layout in itertools.product((80,89,90),(0.,.5,1.),(False,True),("forward","dq_ds_db","dk_dv"),('masked_prefill','decode')):
        fn=kernels._kv_kernel if mode=='dk_dv' else kernels._rows_kernel
        m,n=(1024,1024) if layout=='masked_prefill' else (3,1057)
        values=dict(H=3,M=m,N=n,D=128,BATCH_HEADS=3,DTYPE=1,SCALE=1.,INV_DIVISOR=1/256,
            HAS_SCALE=True,HAS_MASK=layout=='masked_prefill',HAS_BIAS=use_bias,ALPHA=alpha,ANCHOR=512.,CAUSAL=True,OFFSET=n-m,
            BACKWARD=mode=='dq_ds_db',NEED_DQ=mode=='dq_ds_db',NEED_DS=mode=='dq_ds_db',NEED_DB=mode=='dq_ds_db' and use_bias,
            SPLITS=4 if layout=='decode' and mode=='forward' else 1,BLOCK_D=128,BLOCK_M=32,BLOCK_N=32,NEED_DK=True,NEED_DV=True,stride_maskb=n,stride_maskn=1)
        for tensor in ('q','k','v','do'):
            length=m if tensor in ('q','do') else n
            for axis,stride in (('b',3*length*128),('h',length*128),('m' if tensor in ('q','do') else 'n',128)):
                values[f'stride_{tensor}{axis}']=stride
        constants={name:values[name] for i,name in enumerate(fn.arg_names) if i in fn.constexprs}
        signature={name:'constexpr' if i in fn.constexprs else '*fp32' if name in ('Scale','DScale','Bias','DBias') else
                   '*i1' if name=='Mask' else '*i32' if name=='Counts' else '*fp32' if name=='Out' and values['SPLITS']>1 else '*bf16'
                   for i,name in enumerate(fn.arg_names)}
        kernel=triton.compile(ASTSource(fn,signature,constexprs=constants),target=GPUTarget('cuda',sm,32),
                              options=dict(num_warps=4,num_stages=2))
        if sm==89:assert kernel.metadata.shared<=101376
        row=dict(sm=sm,alpha=alpha,bias=use_bias,mode=mode,layout=layout,shared_bytes=kernel.metadata.shared,status='compiled')
        results.append(row);print(json.dumps(row),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--interpreter',action='store_true');choice.add_argument('--compile',action='store_true')
    p.add_argument('--output',type=Path,default=Path('validation/context_kernels.json'))
    a=p.parse_args()
    if a.interpreter and os.environ.get('TRITON_INTERPRET')!='1':p.error('Set TRITON_INTERPRET=1 before starting Python')
    if a.compile and os.environ.get('TRITON_INTERPRET')=='1':p.error('Unset TRITON_INTERPRET for offline compilation')
    torch.set_num_threads(2)
    k=modules()
    if a.compile:results=compile_cases(k)
    else:
        results=[]
        for alpha,scale,mask,bias in itertools.product((0.,.5,1.),(.8,-.5,0.),(False,True),(False,True)):
            row=interpreted(k,7,11,alpha,scale,mask,bias)
            results.append(row);print(json.dumps(row),flush=True)
        for m,n,offset,splits in ((19,19,0,1),(3,1057,1054,4),(9,5,-4,1)):
            row=interpreted(k,m,n,.5,-.7,True,True,splits,offset)
            results.append(row);print(json.dumps(row),flush=True)
        for mask in (False,True):
            row=interpreted(k,5,9,1.,.8,mask,True,causal=False)
            results.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(triton=triton.__version__,mode='compile' if a.compile else 'interpreter',
        count=len(results),results=results,note='No CUDA execution or throughput measurements.'),indent=2)+'\n')


if __name__=='__main__':main()

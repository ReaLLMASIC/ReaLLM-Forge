"""Independent small-tensor precision oracle for the BF16/FP32 Triton tests.

This code is used only by tests/diagnostics. It never changes model arithmetic.
FP64 algebra defines the ideal output/derivatives. Explicit low-precision casts
predict the tensor-core operands. Per-element intervals propagate FP32 rounding
and BF16 cast boundaries; no tolerance is fitted to a measured kernel result.
"""
from dataclasses import dataclass
import json
import math
import torch

U32 = 2.0 ** -24


def gamma32(terms):
    """Standard sequential-sum error bound gamma_n = n*u/(1-n*u)."""
    product = int(terms) * U32
    if not 0 <= product < .1:
        raise ValueError('This diagnostic is intended for small correctness fixtures')
    return product / (1 - product)


def multiply(a, da, b, db):
    value = a * b
    radius = a.abs() * db + abs(b) * da + da * db
    return value, radius + U32 * (value.abs() + radius)


def cast_interval(value, radius, dtype):
    """Rounding is monotone; preserve intervals crossing a BF16 midpoint."""
    middle = value.to(dtype).double()
    low, high = (value-radius).to(dtype).double(), (value+radius).to(dtype).double()
    return middle, torch.maximum((middle-low).abs(), (high-middle).abs())


@dataclass
class Prediction:
    ideal: torch.Tensor
    rounded: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


def stored(ideal, computed, radius, dtype):
    # Account for the final FP32 arithmetic/store before the output dtype cast.
    radius = radius + U32 * (computed.abs()+radius)
    return Prediction(ideal, computed.to(dtype).double(),
                      (computed-radius).to(dtype).double(), (computed+radius).to(dtype).double())


def dot_prediction(ideal, left, radius, right, dtype, scale=1.0):
    center = left @ right
    uncertainty = radius @ right.abs()
    # +16 covers split-cache partial reduction and loop/tile boundaries.
    uncertainty += gamma32(left.shape[-1]+16) * ((left.abs()+radius) @ right.abs())
    scaled = center * scale
    uncertainty = uncertainty * abs(scale) + U32 * (scaled.abs()+uncertainty*abs(scale))
    return stored(ideal, scaled, uncertainty, dtype)


@torch.no_grad()
def precision_oracle(q, k, v, do, scale, bias=None, *, divisor=2.0, alpha=0.0,
                     anchor=8.0, attention_mask=None, causal=True, causal_offset=None):
    """Predict outputs and first derivatives for precisely these input values.

    Input conversion to FP64 happens BEFORE any reference matmul, so ambient TF32
    and reduced-precision PyTorch reduction settings cannot contaminate it.
    BF16/FP32 are supported. Huge/subnormal/overflow fixtures are out of scope.
    """
    if q.dtype not in (torch.bfloat16, torch.float32) or any(x.dtype!=q.dtype for x in (k,v,do)):
        raise ValueError('Use matched BF16 or FP32 fixture tensors')
    if divisor<=0 or anchor<=0 or alpha not in (0.,.5,1.):
        raise ValueError('Invalid divisor, anchor or exponent')
    qf,kf,vf,df=[x.detach().double() for x in (q,k,v,do)]
    g=scale.detach().double() if torch.is_tensor(scale) else float(scale)
    tau=bias.detach().double()[None,:,None,None] if bias is not None else 0.
    batch,heads,m,d=q.shape;n=k.shape[-2]
    offset=n-m if causal_offset is None else causal_offset
    allowed=torch.ones((batch,1,m,n),device=q.device,dtype=torch.bool)
    if causal:
        allowed &= torch.arange(n,device=q.device)[None,:] <= torch.arange(m,device=q.device)[:,None]+offset
    if attention_mask is not None:allowed &= attention_mask[:,None,None,:n]!=0
    # Independent count via the explicit allowed matrix, not the kernel helper.
    factor=(allowed.sum(-1,keepdim=True).double()/anchor).clamp_min(1).pow(-alpha)
    # FP32 reciprocal/rsqrt, count/anchor conversion and clamp boundary rounding.
    factor_error = factor * (8*U32 if alpha else 0.)
    raw=qf @ kf.transpose(-1,-2)
    raw_error=gamma32(d)*(qf.abs() @ kf.abs().transpose(-1,-2))
    score=raw*g-tau
    score_error=abs(g)*raw_error+gamma32(2)*(abs(g)*raw.abs()+abs(tau)+abs(g)*raw_error)
    positive=score.clamp_min(0).masked_fill(~allowed,0)
    low=(score-score_error).clamp_min(0).masked_fill(~allowed,0)
    high=(score+score_error).clamp_min(0).masked_fill(~allowed,0)
    positive_error=torch.maximum(positive-low,high-positive)
    inv=1./divisor
    inv_error=abs(float(torch.tensor(inv,dtype=torch.float32))-inv)
    weights,we=multiply(positive,positive_error,positive,positive_error)
    weights,we=multiply(weights,we,inv,inv_error)
    weights,we=multiply(weights,we,factor,factor_error)
    wr,wre=cast_interval(weights,we,q.dtype)
    dp=df @ vf.transpose(-1,-2)
    dpe=gamma32(d)*(df.abs() @ vf.abs().transpose(-1,-2))
    ds,de=multiply(dp,dpe,2*inv,2*inv_error)
    ds,de=multiply(ds,de,positive,positive_error)
    ds,de=multiply(ds,de,factor,factor_error)
    dr,dre=cast_interval(ds,de,q.dtype)
    result={
        'output':dot_prediction(weights @ vf,wr,wre,vf,q.dtype),
        'dq':dot_prediction((ds @ kf)*g,dr,dre,kf,q.dtype,g),
        'dk':dot_prediction((ds.transpose(-1,-2) @ qf)*g,dr.transpose(-1,-2),dre.transpose(-1,-2),qf,q.dtype,g),
        'dv':dot_prediction(weights.transpose(-1,-2) @ df,wr.transpose(-1,-2),wre.transpose(-1,-2),df,q.dtype),
    }
    product,pe=multiply(ds,de,raw,raw_error)
    center=product.sum()
    error=pe.sum()+gamma32(product.numel()+16)*(product.abs()+pe).sum()
    scale_dtype=scale.dtype if torch.is_tensor(scale) else torch.float32
    result['dscale']=stored(center,center,error,scale_dtype)
    if bias is not None:
        axes=(0,2,3)
        center=-ds.sum(axes)
        error=de.sum(axes)+gamma32(batch*m*n+16)*(ds.abs()+de).sum(axes)
        result['dbias']=stored(center,center,error,bias.dtype)
    return result


def assess(name, actual, prediction, rms_limit=.015):
    """Keep the original 1.5% aggregate limit; require every entry in its interval."""
    got=actual.detach().double()
    want=prediction.ideal
    if got.shape!=want.shape:raise AssertionError(f'{name}: shape {got.shape} != {want.shape}')
    finite=all(bool(torch.isfinite(x).all()) for x in (got,want,prediction.rounded,prediction.lower,prediction.upper))
    if not finite:return dict(name=name,passed=False,error='nonfinite actual or reference')
    error=(got-want).abs()
    reference_rms=want.square().mean().sqrt()
    rms=error.square().mean().sqrt()/(reference_rms+.001)
    # Only FP64 oracle evaluation noise; not a BF16 absolute-error allowance.
    guard=8*torch.finfo(torch.float64).eps*(1+torch.maximum(prediction.lower.abs(),prediction.upper.abs()))
    outside=(got < prediction.lower-guard)|(got > prediction.upper+guard)
    strict=error > (.015+.035*want.abs())
    severity=error/(.015+.035*want.abs())
    flat=int(severity.argmax())
    index=[];remaining=flat
    for dim in reversed(got.shape):index.append(remaining%dim);remaining//=dim
    index=list(reversed(index))
    scalar=lambda x:float(x.reshape(-1)[flat])
    return dict(name=name,passed=bool(rms<rms_limit) and not bool(outside.any()),
        relative_rms=float(rms),rms_limit=rms_limit,max_abs_error=float(error.max()),
        strict_fp32_tolerance_violations=int(strict.sum()),interval_violations=int(outside.sum()),
        max_abs_difference_from_cast_prediction=float((got-prediction.rounded).abs().max()),
        most_discrepant=dict(index=index,actual=scalar(got),ideal_fp64=scalar(want),
            predicted_with_casts=scalar(prediction.rounded),lower=scalar(prediction.lower),upper=scalar(prediction.upper)))


def assert_precision(name, actual, prediction):
    report=assess(name,actual,prediction)
    assert report['passed'],json.dumps(report,indent=2)
    return report

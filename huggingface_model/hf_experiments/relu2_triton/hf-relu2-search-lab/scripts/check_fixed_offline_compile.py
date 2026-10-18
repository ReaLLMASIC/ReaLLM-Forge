"""Compile BF16 head-128 forward/backward for SM89 without executing a GPU.

Run in an environment with Triton installed. The compilation-only process
removes autotuner decorators so no active CUDA driver is required; it compiles
one explicit valid tile per mode. This is not a speed or CUDA accuracy test.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="validation/offline_compilation.json")
    a = p.parse_args()
    original = triton.autotune
    triton.autotune = lambda *args, **kwargs: lambda fn: fn
    try:
        from hf_model.triton_relu2linear_attention import _rows_kernel, _kv_kernel
    finally:
        triton.autotune = original
    results = []
    for threshold in (2, 4, 8, 16):
        for mode, fn in (("forward", _rows_kernel), ("dq_dscale", _rows_kernel), ("dk_dv", _kv_kernel)):
            values = dict(H=3, M=256, N=256, D=128, BATCH_HEADS=3, DTYPE=1, SCALE=1., INV_DIVISOR=1/256,
                THRESHOLD=float(threshold), FIXED8=threshold == 8, HAS_SCALE=True, HAS_MASK=False,
                CAUSAL=True, OFFSET=0, BACKWARD=mode == "dq_dscale", NEED_DQ=mode == "dq_dscale",
                NEED_DS=mode == "dq_dscale", SPLITS=1, BLOCK_D=128, BLOCK_M=32, BLOCK_N=32,
                NEED_DK=True, NEED_DV=True, stride_maskb=256, stride_maskn=1)
            for tensor in ("q", "k", "v", "do"):
                for axis, stride in (("b", 3*256*128), ("h", 256*128), ("m" if tensor in ("q", "do") else "n", 128)):
                    values[f"stride_{tensor}{axis}"] = stride
            constants = {name: values[name] for i, name in enumerate(fn.arg_names) if i in fn.constexprs}
            signature = {name: "constexpr" if i in fn.constexprs else
                         "*fp32" if name in ("Scale", "DScale") else "*i1" if name == "Mask" else "*bf16"
                         for i, name in enumerate(fn.arg_names)}
            kernel = triton.compile(ASTSource(fn, signature, constexprs=constants), target=GPUTarget("cuda", 89, 32),
                                    options=dict(num_warps=4, num_stages=2))
            entry = dict(threshold=threshold, mode=mode, target="sm89", head_dim=128,
                         shared_bytes=kernel.metadata.shared, ptx_bytes=len(kernel.asm["ptx"]), status="compiled")
            assert kernel.metadata.shared <= 101376, entry
            results.append(entry)
            print(json.dumps(entry), flush=True)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(triton_version=triton.__version__, results=results,
        note="Offline compilation only; no CUDA execution, gradient validation or speed measurements."), indent=2) + "\n")


if __name__ == "__main__":
    main()

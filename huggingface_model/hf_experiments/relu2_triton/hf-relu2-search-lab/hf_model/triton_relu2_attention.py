"""Memory-efficient, unnormalized ReLU-squared attention for NVIDIA GPUs.

The operation is ``(relu(scale * Q @ K.T)**2 / divisor) @ V`` with optional
bottom-right causal and key-padding masks. There is deliberately no row-sum
normalization. Tensor-core tiles fuse both matmuls with the activation; neither
forward nor backward writes a quadratic attention matrix to device memory.

Scores, nonlinearities and accumulators use FP32. Tile weights/derivatives are
converted to the input dtype for tensor-core matmuls (FP32 uses TF32x3). Like
fused softmax attention, this does not reproduce eager low-precision rounding
at every intermediate operation. Compare with a high-precision mathematical
reference using dtype-appropriate tolerances, not bitwise equality.

First-order gradients are supported for Q, K, V and a scalar tensor scale.
Backward recomputes tiles, with one owner per dQ or dK/dV tile and no atomics.
The scale derivative uses unscaled QK directly, including negative/zero scales.
"""

import importlib.util
import math
import numbers

import torch
from torch.autograd.function import once_differentiable

TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    # A bounded search covers small/large heads and SM80/SM89/SM90 without
    # an expensive Cartesian product. Triton additionally keys by input dtype.
    _CONFIGS = [
        # TF32x3 large-head backward needs these smaller tiles on SM86/SM89
        # (99 KiB shared-memory limit); even 32x32 can exceed that limit.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ]

    def _prune_configs(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}
        d, dtype = args["D"], args["DTYPE"]
        # Conservative shared-memory/register limits for 24-GB consumer GPUs.
        return [c for c in configs if
                not (c.kwargs["BLOCK_M"] == 16 and not (dtype == 2 and d > 64))
                and not (dtype == 2 and d > 128 and max(c.kwargs.values()) > 32)
                and not (d > 128 and (max(c.kwargs.values()) > 64 or c.num_stages > 2))
                and not (dtype == 2 and (max(c.kwargs.values()) > 64 or c.num_stages > 2))]

    @triton.jit
    def _row_tiles(acc, scale_acc, q, do, K, V, Mask, scale,
                   stride_kn: tl.constexpr, stride_vn: tl.constexpr,
                   stride_maskn: tl.constexpr,
                   M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                   INV_DIVISOR: tl.constexpr, OFFSET: tl.constexpr,
                   rows, cols, dims, lo, hi,
                   HAS_MASK: tl.constexpr, CAUSAL_BAND: tl.constexpr,
                   BACKWARD: tl.constexpr, NEED_DQ: tl.constexpr,
                   NEED_DS: tl.constexpr, BLOCK_N: tl.constexpr):
        for start_n in tl.range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            ns = start_n + cols
            kt = tl.load(K + dims[:, None] + ns[None, :] * stride_kn,
                         (dims[:, None] < D) & (ns[None, :] < N), other=0)
            raw = tl.dot(q, kt, input_precision="tf32x3")
            positive = tl.maximum(raw * scale, 0.0)
            valid = (rows[:, None] < M) & (ns[None, :] < N)
            if CAUSAL_BAND:
                valid = valid & (ns[None, :] <= rows[:, None] + OFFSET)
            if HAS_MASK:
                keep = tl.load(Mask + ns * stride_maskn, ns < N, other=0) != 0
                valid = valid & keep[None, :]
            positive = tl.where(valid, positive, 0.0)
            if BACKWARD:
                vt = tl.load(V + dims[:, None] + ns[None, :] * stride_vn,
                             (dims[:, None] < D) & (ns[None, :] < N), other=0)
                dp = tl.dot(do, vt, input_precision="tf32x3")
                du = dp * (2.0 * INV_DIVISOR) * positive
                if NEED_DQ:
                    # Keep the scale outside the low-precision derivative cast.
                    acc = tl.dot(du.to(q.dtype), tl.trans(kt), acc,
                                 input_precision="tf32x3")
                if NEED_DS:
                    scale_acc += tl.sum(du * raw, 1)
            else:
                v = tl.load(V + ns[:, None] * stride_vn + dims[None, :],
                            (ns[:, None] < N) & (dims[None, :] < D), other=0)
                weights = (positive * positive * INV_DIVISOR).to(q.dtype)
                acc = tl.dot(weights, v, acc, input_precision="tf32x3")
        return acc, scale_acc

    @triton.autotune(
        configs=_CONFIGS,
        key=["M", "N", "D", "BATCH_HEADS", "DTYPE", "CAUSAL", "OFFSET",
             "HAS_MASK", "BACKWARD", "NEED_DQ", "NEED_DS", "SPLITS"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )
    @triton.jit
    def _rows_kernel(Q, K, V, Out, DO, DScale, Scale, Mask,
                     stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr,
                     stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
                     stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr,
                     stride_dob: tl.constexpr, stride_doh: tl.constexpr, stride_dom: tl.constexpr,
                     stride_maskb: tl.constexpr, stride_maskn: tl.constexpr,
                     H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                     BATCH_HEADS: tl.constexpr, DTYPE: tl.constexpr,
                     SCALE: tl.constexpr, INV_DIVISOR: tl.constexpr,
                     HAS_SCALE: tl.constexpr, HAS_MASK: tl.constexpr,
                     CAUSAL: tl.constexpr, OFFSET: tl.constexpr,
                     BACKWARD: tl.constexpr, NEED_DQ: tl.constexpr, NEED_DS: tl.constexpr,
                     SPLITS: tl.constexpr, BLOCK_D: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        row_start = tl.program_id(0) * BLOCK_M
        bh = tl.program_id(1)
        batch, head = bh // H, bh % H
        # Cast before multiplication to permit large, strided tensor addresses.
        batch, head = batch.to(tl.int64), head.to(tl.int64)
        Q += batch * stride_qb + head * stride_qh
        K += batch * stride_kb + head * stride_kh
        V += batch * stride_vb + head * stride_vh
        if HAS_MASK:
            Mask += batch * stride_maskb
        rows = row_start + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        q = tl.load(Q + rows[:, None] * stride_qm + dims[None, :],
                    (rows[:, None] < M) & (dims[None, :] < D), other=0)
        if HAS_SCALE:
            scale = tl.load(Scale).to(tl.float32)
        else:
            scale = SCALE
        if BACKWARD:
            DO += batch * stride_dob + head * stride_doh
            do = tl.load(DO + rows[:, None] * stride_dom + dims[None, :],
                         (rows[:, None] < M) & (dims[None, :] < D), other=0)
        else:
            do = q  # Eliminated by the compile-time BACKWARD branch.
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        scale_acc = tl.zeros((BLOCK_M,), tl.float32)
        split = tl.program_id(2)
        split_width = tl.cdiv(N, SPLITS * BLOCK_N) * BLOCK_N
        lo = split * split_width
        hi = tl.minimum(N, lo + split_width)
        if CAUSAL:
            hi = tl.minimum(hi, tl.maximum(0, row_start + BLOCK_M + OFFSET))
            # Keys strictly below this aligned boundary need no causal compare.
            full_end = tl.minimum(hi, tl.maximum(lo, row_start + OFFSET + 1))
            full_end = tl.maximum(lo, full_end // BLOCK_N * BLOCK_N)
            acc, scale_acc = _row_tiles(
                acc, scale_acc, q, do, K, V, Mask, scale,
                stride_kn, stride_vn, stride_maskn, M, N, D, INV_DIVISOR, OFFSET,
                rows, cols, dims, lo, full_end, HAS_MASK, False,
                BACKWARD, NEED_DQ, NEED_DS, BLOCK_N)
            acc, scale_acc = _row_tiles(
                acc, scale_acc, q, do, K, V, Mask, scale,
                stride_kn, stride_vn, stride_maskn, M, N, D, INV_DIVISOR, OFFSET,
                rows, cols, dims, full_end, hi, HAS_MASK, True,
                BACKWARD, NEED_DQ, NEED_DS, BLOCK_N)
        else:
            acc, scale_acc = _row_tiles(
                acc, scale_acc, q, do, K, V, Mask, scale,
                stride_kn, stride_vn, stride_maskn, M, N, D, INV_DIVISOR, OFFSET,
                rows, cols, dims, lo, hi, HAS_MASK, False,
                BACKWARD, NEED_DQ, NEED_DS, BLOCK_N)
        if not BACKWARD or NEED_DQ:
            if BACKWARD:
                acc *= scale
            offsets = ((split.to(tl.int64) * BATCH_HEADS + bh) * M + rows[:, None]) * D + dims[None, :]
            tl.store(Out + offsets, acc, (rows[:, None] < M) & (dims[None, :] < D))
        if NEED_DS:
            tl.store(DScale + bh.to(tl.int64) * M + rows, scale_acc, rows < M)

    @triton.jit
    def _kv_tiles(dk, dv, k, v, Q, DO, keep, scale,
                  stride_qm: tl.constexpr, stride_dom: tl.constexpr,
                  M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                  INV_DIVISOR: tl.constexpr, OFFSET: tl.constexpr,
                  rows, cols, dims, lo, hi,
                  CAUSAL_BAND: tl.constexpr, NEED_DK: tl.constexpr,
                  NEED_DV: tl.constexpr, BLOCK_M: tl.constexpr):
        for start_m in tl.range(lo, hi, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            ms = start_m + rows
            qt = tl.load(Q + dims[:, None] + ms[None, :] * stride_qm,
                         (dims[:, None] < D) & (ms[None, :] < M), other=0)
            raw = tl.dot(k, qt, input_precision="tf32x3")
            positive = tl.maximum(raw * scale, 0.0)
            valid = keep[:, None] & (ms[None, :] < M)
            if CAUSAL_BAND:
                valid = valid & (cols[:, None] <= ms[None, :] + OFFSET)
            positive = tl.where(valid, positive, 0.0)
            do = tl.load(DO + ms[:, None] * stride_dom + dims[None, :],
                         (ms[:, None] < M) & (dims[None, :] < D), other=0)
            if NEED_DV:
                weights = (positive * positive * INV_DIVISOR).to(k.dtype)
                dv = tl.dot(weights, do, dv, input_precision="tf32x3")
            if NEED_DK:
                dp = tl.dot(v, tl.trans(do), input_precision="tf32x3")
                du = dp * (2.0 * INV_DIVISOR) * positive
                dk = tl.dot(du.to(k.dtype), tl.trans(qt), dk, input_precision="tf32x3")
        return dk, dv

    @triton.autotune(
        configs=_CONFIGS,
        key=["M", "N", "D", "BATCH_HEADS", "DTYPE", "CAUSAL", "OFFSET",
             "HAS_MASK", "NEED_DK", "NEED_DV"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )
    @triton.jit
    def _kv_kernel(Q, K, V, DO, DK, DV, Scale, Mask,
                   stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr,
                   stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
                   stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr,
                   stride_dob: tl.constexpr, stride_doh: tl.constexpr, stride_dom: tl.constexpr,
                   stride_maskb: tl.constexpr, stride_maskn: tl.constexpr,
                   H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                   BATCH_HEADS: tl.constexpr, DTYPE: tl.constexpr,
                   SCALE: tl.constexpr, INV_DIVISOR: tl.constexpr,
                   HAS_SCALE: tl.constexpr, HAS_MASK: tl.constexpr,
                   CAUSAL: tl.constexpr, OFFSET: tl.constexpr,
                   NEED_DK: tl.constexpr, NEED_DV: tl.constexpr,
                   BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        col_start = tl.program_id(0) * BLOCK_N
        bh = tl.program_id(1)
        batch, head = (bh // H).to(tl.int64), (bh % H).to(tl.int64)
        Q += batch * stride_qb + head * stride_qh
        K += batch * stride_kb + head * stride_kh
        V += batch * stride_vb + head * stride_vh
        DO += batch * stride_dob + head * stride_doh
        rows = tl.arange(0, BLOCK_M)
        cols = col_start + tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        k = tl.load(K + cols[:, None] * stride_kn + dims[None, :],
                    (cols[:, None] < N) & (dims[None, :] < D), other=0)
        if NEED_DK:
            v = tl.load(V + cols[:, None] * stride_vn + dims[None, :],
                        (cols[:, None] < N) & (dims[None, :] < D), other=0)
        else:
            v = k
        keep = cols < N
        if HAS_MASK:
            keep = keep & (tl.load(Mask + batch * stride_maskb + cols * stride_maskn,
                                   cols < N, other=0) != 0)
        if HAS_SCALE:
            scale = tl.load(Scale).to(tl.float32)
        else:
            scale = SCALE
        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        if CAUSAL:
            lo = tl.maximum(0, col_start - OFFSET) // BLOCK_M * BLOCK_M
            # Only the diagonal band requires the row/key causal comparison.
            full_start = tl.cdiv(tl.maximum(0, col_start + BLOCK_N - 1 - OFFSET), BLOCK_M) * BLOCK_M
            full_start = tl.minimum(M, full_start)
            dk, dv = _kv_tiles(
                dk, dv, k, v, Q, DO, keep, scale, stride_qm, stride_dom,
                M, N, D, INV_DIVISOR, OFFSET, rows, cols, dims, lo, full_start,
                True, NEED_DK, NEED_DV, BLOCK_M)
            dk, dv = _kv_tiles(
                dk, dv, k, v, Q, DO, keep, scale, stride_qm, stride_dom,
                M, N, D, INV_DIVISOR, OFFSET, rows, cols, dims, full_start, M,
                False, NEED_DK, NEED_DV, BLOCK_M)
        else:
            dk, dv = _kv_tiles(
                dk, dv, k, v, Q, DO, keep, scale, stride_qm, stride_dom,
                M, N, D, INV_DIVISOR, OFFSET, rows, cols, dims, 0, M,
                False, NEED_DK, NEED_DV, BLOCK_M)
        offsets = (bh.to(tl.int64) * N + cols[:, None]) * D + dims[None, :]
        valid = (cols[:, None] < N) & (dims[None, :] < D)
        if NEED_DK:
            tl.store(DK + offsets, dk * scale, valid)
        if NEED_DV:
            tl.store(DV + offsets, dv, valid)

    @triton.jit
    def _reduce_splits(Partials, Out, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                       BLOCK: tl.constexpr, BLOCK_SPLITS: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        splits = tl.arange(0, BLOCK_SPLITS)
        values = tl.load(Partials + splits[:, None] * SIZE + offsets[None, :],
                         (splits[:, None] < SPLITS) & (offsets[None, :] < SIZE), other=0)
        tl.store(Out + offsets, tl.sum(values, 0), offsets < SIZE)


def can_use_triton_relu2_attention(q, k, v, attention_mask=None):
    """Return False for unsupported devices, shapes, dtypes or memory layouts.

    Arbitrary outer strides (including expanded KV heads) are accepted; the
    feature dimension must have stride one. Float masks, like integer/bool
    masks, mean "keep nonzero"; additive attention bias is not supported.
    """
    if not TRITON_AVAILABLE or not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        return False
    if not all(t.is_cuda and t.ndim == 4 for t in (q, k, v)):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if any(t.device != q.device or t.dtype != q.dtype or t.stride(-1) != 1 for t in (k, v)):
        return False
    if q.stride(-1) != 1 or any(size <= 0 for t in (q, k, v) for size in t.shape):
        return False
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        return False
    if q.shape[-1] > 256:
        return False
    # This implementation uses NVIDIA tensor cores and TF32x3 for FP32.
    if torch.version.hip is not None or torch.cuda.get_device_capability(q.device)[0] < 8:
        return False
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            return False
        if attention_mask.device != q.device or attention_mask.ndim != 2:
            return False
        if attention_mask.shape[0] != q.shape[0] or attention_mask.shape[1] < k.shape[2]:
            return False
        if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16,
                                       torch.int32, torch.int64, torch.float16,
                                       torch.bfloat16, torch.float32, torch.float64):
            return False
    return True


def _launch_options(q, k, v, do, scale, divisor, causal, offset, mask):
    b, h, m, d = q.shape
    return dict(
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2),
        stride_dob=do.stride(0), stride_doh=do.stride(1), stride_dom=do.stride(2),
        stride_maskb=mask.stride(0) if mask is not None else 0,
        stride_maskn=mask.stride(1) if mask is not None else 0,
        H=h, M=m, N=k.shape[2], D=d, BATCH_HEADS=b * h,
        DTYPE={torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[q.dtype],
        SCALE=scale if not isinstance(scale, torch.Tensor) else 1.0,
        INV_DIVISOR=1.0 / divisor,
        HAS_SCALE=isinstance(scale, torch.Tensor), HAS_MASK=mask is not None,
        CAUSAL=causal, OFFSET=offset, BLOCK_D=max(16, triton.next_power_of_2(d)),
    )


class _TritonReLU2Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, divisor, causal, causal_offset, attention_mask):
        b, h, m, d = q.shape
        n = k.shape[2]
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        options = _launch_options(q, k, v, q, scale, divisor, causal, causal_offset, attention_mask)
        # A single query/head otherwise launches only one CTA. Partition the
        # KV sequence for cached decoding, then sum FP32 partial outputs.
        splits = min(16, triton.cdiv(128, b * h), n // 512) if m <= 16 and n >= 1024 else 1
        target = torch.empty((splits, b, h, m, d), dtype=torch.float32, device=q.device) if splits > 1 else out
        _rows_kernel[lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), b * h, splits)](
            q, k, v, target, q, q,
            scale if isinstance(scale, torch.Tensor) else q,
            attention_mask if attention_mask is not None else q,
            **options, BACKWARD=False, NEED_DQ=False, NEED_DS=False, SPLITS=splits,
        )
        if splits > 1:
            _reduce_splits[(triton.cdiv(out.numel(), 128),)](
                target, out, SIZE=out.numel(), SPLITS=splits,
                BLOCK=128, BLOCK_SPLITS=triton.next_power_of_2(splits), num_warps=4)
        ctx.save_for_backward(q, k, v, scale if isinstance(scale, torch.Tensor) else None, attention_mask)
        ctx.scalar_scale = None if isinstance(scale, torch.Tensor) else scale
        ctx.divisor, ctx.causal, ctx.causal_offset = divisor, causal, causal_offset
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, tensor_scale, mask = ctx.saved_tensors
        scale = tensor_scale if tensor_scale is not None else ctx.scalar_scale
        # Expanded/sliced upstream gradients are common (e.g. output.sum()).
        if grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        need_q, need_k, need_v, need_s = ctx.needs_input_grad[:4]
        need_s = need_s and tensor_scale is not None
        dq = torch.empty(q.shape, dtype=q.dtype, device=q.device) if need_q else None
        dk = torch.empty(k.shape, dtype=k.dtype, device=k.device) if need_k else None
        dv = torch.empty(v.shape, dtype=v.dtype, device=v.device) if need_v else None
        ds_parts = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device) if need_s else None
        options = _launch_options(q, k, v, grad_output, scale, ctx.divisor, ctx.causal, ctx.causal_offset, mask)
        scale_ptr = tensor_scale if tensor_scale is not None else q
        mask_ptr = mask if mask is not None else q
        b, h, m, _ = q.shape
        # Backward may run after the caller changes the current CUDA device.
        with torch.cuda.device(q.device):
            if need_q or need_s:
                _rows_kernel[lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), b * h, 1)](
                    q, k, v, dq if need_q else q, grad_output, ds_parts if need_s else q,
                    scale_ptr, mask_ptr, **options,
                    BACKWARD=True, NEED_DQ=need_q, NEED_DS=need_s, SPLITS=1,
                )
            if need_k or need_v:
                _kv_kernel[lambda meta: (triton.cdiv(k.shape[2], meta["BLOCK_N"]), b * h)](
                    q, k, v, grad_output, dk if need_k else k, dv if need_v else v,
                    scale_ptr, mask_ptr, **options, NEED_DK=need_k, NEED_DV=need_v,
                )
        ds = ds_parts.sum().to(tensor_scale.dtype).reshape(tensor_scale.shape) if need_s else None
        return dq, dk, dv, ds, None, None, None, None


def triton_relu2_attention(q, k, v, scale=None, divisor=256.0, causal=True,
                          causal_offset=None, attention_mask=None):
    """Compute tiled ReLU² attention, preserving gradients without an N² buffer.

    Args:
        q, k, v: Same dtype/device; [B,H,M,D], [B,H,N,D], [B,H,N,D].
            D <= 256, unit feature stride; arbitrary outer strides are allowed.
        scale: Finite Python real or one-element CUDA FP16/BF16/FP32 tensor.
            Defaults to 1/sqrt(D). Tensor scales can be learned, signed or zero.
        divisor: Positive finite Python real. No gradient is defined for it.
        causal: Whether key n must satisfy n <= m + causal_offset.
        causal_offset: Defaults to N-M (bottom-right alignment for KV caching).
        attention_mask: Optional [B, >=N] numeric/bool key mask; nonzero keeps.

    Requires NVIDIA SM80+ and Triton. Unsupported calls raise ValueError; call
    ``can_use_triton_relu2_attention`` when implementing automatic fallback.
    Autotuning runs on the first call for each shape/mode/dtype/mask combination;
    warm both forward and backward before timing. Higher derivatives, attention
    dropout, additive biases and unequal Q/K/V feature sizes are unsupported.
    """
    if not can_use_triton_relu2_attention(q, k, v, attention_mask):
        raise ValueError("Fused ReLU2 attention requires Triton, NVIDIA SM80+ CUDA tensors "
                         "of equal FP16/BF16/FP32 dtype, matching B/H/D (D <= 256), "
                         "unit feature strides and an optional [B, >=N] key mask")
    if not isinstance(divisor, numbers.Real) or isinstance(divisor, bool) or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be a positive finite Python real")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    elif isinstance(scale, torch.Tensor):
        if scale.device != q.device or scale.numel() != 1 or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("scale must be one FP16/BF16/FP32 CUDA value on the Q/K/V device")
    elif not isinstance(scale, numbers.Real) or isinstance(scale, bool) or not math.isfinite(scale):
        raise ValueError("scale must be a finite Python real or a one-element CUDA tensor")
    else:
        scale = float(scale)
    if not isinstance(causal, bool):
        raise ValueError("causal must be a bool")
    if causal_offset is None:
        causal_offset = k.shape[2] - q.shape[2]
    if not isinstance(causal_offset, numbers.Integral) or isinstance(causal_offset, bool):
        raise ValueError("causal_offset must be an integer")
    # Triton launches and autotuning use the current CUDA device, which can
    # differ from q.device in applications placing models on several GPUs.
    with torch.cuda.device(q.device):
        return _TritonReLU2Attention.apply(q, k, v, scale, float(divisor), causal,
                                         int(causal_offset), attention_mask)

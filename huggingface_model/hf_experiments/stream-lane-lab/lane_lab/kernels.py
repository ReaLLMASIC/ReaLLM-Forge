"""Small differentiable Triton primitives for the per-column geometries.

The transformer and dense projections use PyTorch/SDPA/BLAS. Triton accelerates
the gather/interpolation operations below, including their backward passes.
Angles and SLERP coefficients are computed by ordinary PyTorch autograd.

``auto`` uses Triton on supported CUDA tensors and PyTorch otherwise. Selecting
``triton`` explicitly never silently falls back. CUDA backward uses floating
point atomics, so its summation order is not deterministic. No CUDA speed claim
is implied by the existence of these kernels: benchmark on the target machine.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _dispatch(tensor: torch.Tensor, backend: str, *others: torch.Tensor) -> bool:
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"Unknown backend {backend!r}; expected auto, torch, or triton")
    if backend == "torch":
        return False
    reason = None
    if tensor.device.type != "cuda":
        reason = "a CUDA tensor is required"
    elif triton is None:
        reason = "Triton is not installed"
    elif any(t.device != tensor.device for t in others):
        reason = "all tensors must be on the same CUDA device"
    elif any(t.is_floating_point() and t.dtype not in {torch.float16, torch.bfloat16, torch.float32}
             for t in (tensor, *others)):
        reason = "only float32, float16, and bfloat16 are supported"
    elif tensor.numel() == 0:
        reason = "empty tensors use the PyTorch implementation"
    elif torch.is_grad_enabled() and torch.are_deterministic_algorithms_enabled() and any(
        t.requires_grad for t in (tensor, *others)
    ):
        reason = "Triton backward uses nondeterministic atomics; select torch for deterministic training"
    if reason:
        if backend == "triton":
            raise RuntimeError(f"Triton backend unavailable: {reason}")
        return False
    return True


def _indices(indices: torch.Tensor, tensor: torch.Tensor) -> None:
    if indices.dtype != torch.long:
        raise TypeError("Indices must have dtype torch.long")
    if indices.device != tensor.device:
        raise ValueError("Indices and values must be on the same device")


if triton is not None:

    @triton.jit
    def _gather_forward(Table, Idx, Out, N: tl.constexpr, D: tl.constexpr,
                        BLOCK: tl.constexpr):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, dim = pos // D, pos % D
        k = tl.load(Idx + row, row < N, 0)
        val = tl.load(Table + k * D + dim, pos < N * D, 0)
        tl.store(Out + pos, val, pos < N * D)

    @triton.jit
    def _gather_backward(Grad, Idx, GradTable, N: tl.constexpr, D: tl.constexpr,
                         BLOCK: tl.constexpr):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, dim = pos // D, pos % D
        k = tl.load(Idx + row, row < N, 0)
        g = tl.load(Grad + pos, pos < N * D, 0).to(tl.float32)
        tl.atomic_add(GradTable + k * D + dim, g, pos < N * D, sem="relaxed")

    @triton.jit
    def _weighted_forward(Table, Left, Right, A, B, Out,
                          N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        dim = tl.arange(0, BLOCK)
        left, right = tl.load(Left + row), tl.load(Right + row)
        a, b = tl.load(A + row).to(tl.float32), tl.load(B + row).to(tl.float32)
        x = tl.load(Table + left * D + dim, dim < D, 0).to(tl.float32)
        y = tl.load(Table + right * D + dim, dim < D, 0).to(tl.float32)
        tl.store(Out + row * D + dim, a * x + b * y, dim < D)

    @triton.jit
    def _weighted_backward(Grad, Table, Left, Right, A, B, GradTable, GradA, GradB,
                           N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        dim = tl.arange(0, BLOCK)
        left, right = tl.load(Left + row), tl.load(Right + row)
        a, b = tl.load(A + row).to(tl.float32), tl.load(B + row).to(tl.float32)
        x = tl.load(Table + left * D + dim, dim < D, 0).to(tl.float32)
        y = tl.load(Table + right * D + dim, dim < D, 0).to(tl.float32)
        g = tl.load(Grad + row * D + dim, dim < D, 0).to(tl.float32)
        tl.atomic_add(GradTable + left * D + dim, a * g, dim < D, sem="relaxed")
        tl.atomic_add(GradTable + right * D + dim, b * g, dim < D, sem="relaxed")
        tl.store(GradA + row, tl.sum(g * x, axis=0))
        tl.store(GradB + row, tl.sum(g * y, axis=0))

    @triton.jit
    def _blend_forward(Scores, Left, Right, A, B, Out, N: tl.constexpr,
                       K: tl.constexpr, Q: tl.constexpr, BLOCK: tl.constexpr):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, q = pos // Q, pos % Q
        left, right = tl.load(Left + q), tl.load(Right + q)
        a, b = tl.load(A + q).to(tl.float32), tl.load(B + q).to(tl.float32)
        x = tl.load(Scores + row * K + left, row < N, 0).to(tl.float32)
        y = tl.load(Scores + row * K + right, row < N, 0).to(tl.float32)
        tl.store(Out + pos, a * x + b * y, row < N)

    @triton.jit
    def _blend_backward(Grad, Scores, Left, Right, A, B, GradScores, GradA, GradB,
                        N: tl.constexpr, K: tl.constexpr, Q: tl.constexpr,
                        BLOCK: tl.constexpr):
        pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, q = pos // Q, pos % Q
        left, right = tl.load(Left + q), tl.load(Right + q)
        a, b = tl.load(A + q).to(tl.float32), tl.load(B + q).to(tl.float32)
        x = tl.load(Scores + row * K + left, row < N, 0).to(tl.float32)
        y = tl.load(Scores + row * K + right, row < N, 0).to(tl.float32)
        g = tl.load(Grad + pos, row < N, 0).to(tl.float32)
        tl.atomic_add(GradScores + row * K + left, a * g, row < N, sem="relaxed")
        tl.atomic_add(GradScores + row * K + right, b * g, row < N, sem="relaxed")
        tl.atomic_add(GradA + q, x * g, row < N, sem="relaxed")
        tl.atomic_add(GradB + q, y * g, row < N, sem="relaxed")


class _Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, indices):
        ctx.save_for_backward(indices)
        ctx.table_shape, ctx.table_dtype = table.shape, table.dtype
        n, d = indices.numel(), table.shape[1]
        out = torch.empty((*indices.shape, d), device=table.device, dtype=table.dtype)
        with torch.cuda.device(table.device):
            _gather_forward[(triton.cdiv(n * d, 256),)](table, indices, out, n, d, 256)
        return out

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        n, d = indices.numel(), ctx.table_shape[1]
        dt = torch.zeros(ctx.table_shape, device=grad.device, dtype=torch.float32)
        with torch.cuda.device(grad.device):
            _gather_backward[(triton.cdiv(n * d, 256),)](grad.contiguous(), indices, dt, n, d, 256)
        return dt.to(ctx.table_dtype), None


class _WeightedLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, table, left, right, a, b):
        ctx.save_for_backward(table, left, right, a, b)
        n, d = left.numel(), table.shape[1]
        dtype = torch.promote_types(table.dtype, torch.promote_types(a.dtype, b.dtype))
        out = torch.empty((*left.shape, d), device=table.device, dtype=dtype)
        with torch.cuda.device(table.device):
            _weighted_forward[(n,)](table, left, right, a, b, out, n, d, triton.next_power_of_2(d))
        return out

    @staticmethod
    def backward(ctx, grad):
        table, left, right, a, b = ctx.saved_tensors
        n, d = left.numel(), table.shape[1]
        dt = torch.zeros_like(table, dtype=torch.float32)
        da, db = torch.empty_like(a, dtype=torch.float32), torch.empty_like(b, dtype=torch.float32)
        with torch.cuda.device(table.device):
            _weighted_backward[(n,)](grad.contiguous(), table, left, right, a, b, dt, da, db,
                                     n, d, triton.next_power_of_2(d))
        return dt.to(table.dtype), None, None, da.to(a.dtype), db.to(b.dtype)


class _BlendScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, left, right, a, b):
        ctx.save_for_backward(scores, left, right, a, b)
        k, q = scores.shape[-1], left.numel()
        n = scores.numel() // k
        dtype = torch.promote_types(scores.dtype, torch.promote_types(a.dtype, b.dtype))
        out = torch.empty((*scores.shape[:-1], q), device=scores.device, dtype=dtype)
        with torch.cuda.device(scores.device):
            _blend_forward[(triton.cdiv(n * q, 256),)](scores, left, right, a, b, out, n, k, q, 256)
        return out

    @staticmethod
    def backward(ctx, grad):
        scores, left, right, a, b = ctx.saved_tensors
        k, q = scores.shape[-1], left.numel()
        n = scores.numel() // k
        ds = torch.zeros_like(scores, dtype=torch.float32)
        da, db = torch.zeros_like(a, dtype=torch.float32), torch.zeros_like(b, dtype=torch.float32)
        with torch.cuda.device(scores.device):
            _blend_backward[(triton.cdiv(n * q, 256),)](
                grad.contiguous(), scores, left, right, a, b, ds, da, db, n, k, q, 256)
        return ds.to(scores.dtype), None, None, da.to(a.dtype), db.to(b.dtype)


def lookup(table: torch.Tensor, index: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Gather ``table[index]``. Indices must be in [0, table.shape[0])."""
    if table.ndim != 2 or not table.is_floating_point():
        raise ValueError("table must be a floating point [K,D] tensor")
    _indices(index, table)
    use_triton = _dispatch(table, backend, index)
    if index.numel() == 0:
        return table[index]
    if use_triton:
        return _Lookup.apply(table.contiguous(), index.contiguous())
    return table[index]


def weighted_lookup(table: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                    a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Return ``a[...,None]*table[left] + b[...,None]*table[right]``.

    ``left``, ``right``, ``a``, and ``b`` must have the same shape. Coefficients
    remain differentiable; gradients are propagated to both coefficients and
    the shared knot table. Repeated knots accumulate their gradients.
    """
    if table.ndim != 2 or not table.is_floating_point():
        raise ValueError("table must be a floating point [K,D] tensor")
    _indices(left, table)
    _indices(right, table)
    if any(x.shape != left.shape for x in (right, a, b)):
        raise ValueError("left, right, a, and b must have the same shape")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Interpolation coefficients must be floating point")
    use_triton = _dispatch(table, backend, left, right, a, b)
    if table.shape[1] > 4096:
        if backend == "triton":
            raise RuntimeError("Triton weighted lookup supports D <= 4096")
        use_triton = False
    if use_triton and left.numel():
        return _WeightedLookup.apply(*(x.contiguous() for x in (table, left, right, a, b)))
    return table[left] * a[..., None] + table[right] * b[..., None]


def blend_scores(scores: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Expand K knot projections to Q class logits without a Q-by-D table.

    ``scores`` has shape [...,K]. Each of ``left,right,a,b`` has shape [Q].
    The result has shape [...,Q]. Both coefficient vectors receive gradients.
    """
    if scores.ndim < 1 or not scores.is_floating_point():
        raise ValueError("scores must be floating point with a final knot dimension")
    _indices(left, scores)
    _indices(right, scores)
    if left.ndim != 1 or any(x.shape != left.shape for x in (right, a, b)):
        raise ValueError("left, right, a, and b must have the same one-dimensional shape")
    if not a.is_floating_point() or not b.is_floating_point():
        raise TypeError("Interpolation coefficients must be floating point")
    use_triton = _dispatch(scores, backend, left, right, a, b)
    if use_triton and left.numel():
        return _BlendScores.apply(*(x.contiguous() for x in (scores, left, right, a, b)))
    return scores[..., left] * a + scores[..., right] * b

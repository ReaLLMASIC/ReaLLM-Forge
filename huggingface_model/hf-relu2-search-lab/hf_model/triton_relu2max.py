"""Optional Triton accelerator for nanoGPT's elementwise ReLU2Max."""

import importlib.util

import torch


TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _relu2max_forward_kernel(x, output, size, inv_divisor, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        values = tl.load(x + offsets, mask=mask)
        positive = tl.maximum(values, 0.0)
        tl.store(output + offsets, positive * positive * inv_divisor, mask=mask)

    @triton.jit
    def _relu2max_backward_kernel(x, grad_output, grad_input, size, inv_divisor, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        values = tl.load(x + offsets, mask=mask)
        gradients = tl.load(grad_output + offsets, mask=mask)
        derivative = tl.where(values > 0.0, 2.0 * values * inv_divisor, 0.0)
        tl.store(grad_input + offsets, gradients * derivative, mask=mask)


class _TritonReLU2Max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, divisor):
        values = values.contiguous()
        output = torch.empty_like(values)
        size = values.numel()
        block_size = 256
        _relu2max_forward_kernel[(triton.cdiv(size, block_size),)](
            values, output, size=size, inv_divisor=1.0 / divisor, BLOCK_SIZE=block_size
        )
        ctx.save_for_backward(values)
        ctx.divisor = divisor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (values,) = ctx.saved_tensors
        grad_input = torch.empty_like(values)
        size = values.numel()
        block_size = 256
        _relu2max_backward_kernel[(triton.cdiv(size, block_size),)](
            values,
            grad_output.contiguous(),
            grad_input,
            size=size,
            inv_divisor=1.0 / ctx.divisor,
            BLOCK_SIZE=block_size,
        )
        return grad_input, None


def triton_relu2max(values, divisor):
    """Compute ``relu(values) ** 2 / divisor`` with a fused Triton kernel."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; use accelerator='torch' or install Triton")
    if not values.is_cuda:
        raise RuntimeError("The Triton ReLU2Max accelerator requires a CUDA tensor")
    return _TritonReLU2Max.apply(values, divisor)


def can_use_triton_relu2max(values):
    """Whether the optional kernel can execute for this tensor."""
    return TRITON_AVAILABLE and values.is_cuda and values.dtype in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }

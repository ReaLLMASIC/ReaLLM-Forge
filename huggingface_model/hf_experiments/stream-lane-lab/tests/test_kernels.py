"""CPU reference checks and the target-GPU parity gate (skipped without CUDA)."""

import pytest
import torch

from lane_lab.kernels import blend_scores, lookup, triton_available, weighted_lookup


def _layout(device, k=5, q=17):
    position = torch.linspace(0, k - 1, q, device=device)
    left = position.floor().long().clamp_max(k - 2)
    right = left + 1
    u = position - left
    theta = torch.linspace(0.2, 2.8, k - 1, device=device)[left]
    return left, right, torch.sin((1 - u) * theta) / theta.sin(), torch.sin(u * theta) / theta.sin()


def test_cpu_reference_gradcheck():
    torch.manual_seed(20)
    table = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    left, right, a, b = _layout("cpu")
    a = a.double().requires_grad_()
    b = b.double().requires_grad_()
    assert torch.autograd.gradcheck(lambda w, x, y: weighted_lookup(w, left, right, x, y, "auto"),
                                    (table, a, b))
    scores = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s, x, y: blend_scores(s, left, right, x, y, "torch"),
                                    (scores, a, b))


def test_factorized_matches_dense_and_endpoints():
    torch.manual_seed(21)
    table = torch.randn(5, 7, requires_grad=True)
    hidden = torch.randn(2, 3, 7, requires_grad=True)
    left, right, a, b = _layout("cpu")
    dense = weighted_lookup(table, left, right, a, b, "torch")
    torch.testing.assert_close(dense[0], table[0])
    torch.testing.assert_close(dense[-1], table[-1])
    # Interior knots are exact shared endpoints at q=4,8,12.
    torch.testing.assert_close(dense[::4], table)
    actual = blend_scores(hidden @ table.T, left, right, a, b, "auto")
    torch.testing.assert_close(actual, hidden @ dense.T)
    index = torch.tensor([[0, 2], [2, 4]])
    torch.testing.assert_close(lookup(table, index, "auto"), table[index])
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        lookup(table, index, "triton")


@pytest.mark.skipif(not triton_available(), reason="Requires CUDA and Triton; run on the target GPU")
@pytest.mark.parametrize("k", [2, 3, 5, 9])
@pytest.mark.parametrize("operation", ["lookup", "weighted", "blend"])
def test_cuda_forward_and_backward(k, operation):
    torch.manual_seed(22)
    # Transposes/slices exercise noncontiguous input handling.
    table = torch.randn(33, k, device="cuda").T.detach().requires_grad_()
    left, right, a, b = _layout("cuda", k=k, q=65)
    a, b = a.detach().requires_grad_(), b.detach().requires_grad_()
    if operation == "lookup":
        index = torch.randint(k, (7, 11), device="cuda").T
        args = (table,)
        fn = lambda backend: lookup(table, index, backend)
    elif operation == "weighted":
        args = (table, a, b)
        fn = lambda backend: weighted_lookup(table, left, right, a, b, backend)
    else:
        scores = torch.randn(2, k, 7, device="cuda").transpose(1, 2).detach().requires_grad_()
        args = (scores, a, b)
        fn = lambda backend: blend_scores(scores, left, right, a, b, backend)
    expected, actual = fn("torch"), fn("triton")
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    grad = torch.randn_like(expected)
    eg = torch.autograd.grad(expected, args, grad)
    ag = torch.autograd.grad(actual, args, grad)
    for x, y in zip(ag, eg):
        torch.testing.assert_close(x, y, atol=8e-5, rtol=8e-5)


@pytest.mark.skipif(not triton_available(), reason="Requires CUDA and Triton")
def test_explicit_triton_rejects_float64_and_auto_falls_back():
    table = torch.randn(3, 7, device="cuda", dtype=torch.float64)
    index = torch.tensor([0, 2], device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        lookup(table, index, "triton")
    torch.testing.assert_close(lookup(table, index, "auto"), table[index])

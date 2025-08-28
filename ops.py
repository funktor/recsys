from . import _C
import torch

__all__ = ['mysoftmax_cpu', 'mysoftmax_gpu']

def mysoftmax_cpu(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.extension_cpp.mysoftmax_cpu.default(a.contiguous())

def mysoftmax_gpu(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.extension_cpp.mysoftmax_gpu.default(a.contiguous())

@torch.library.register_fake("extension_cpp::mysoftmax_cpu")
def _(a:torch.Tensor):
    torch._check(a.dtype == torch.float)
    torch._check(a.device == 'cpu')
    return torch.empty_like(a)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mysoftmax_gpu")
def _(a:torch.Tensor):
    torch._check(a.dtype == torch.float)
    torch._check(a.device == 'cuda:0')
    return torch.empty_like(a)
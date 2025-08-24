import torch

__all__ = ['mysoftmax']

torch.ops.load_library("/Users/amondal/recsys/.venv/lib/python3.13/site-packages/extension_cpp/_C.cpython-313-darwin.so")

def mysoftmax(a: torch.Tensor) -> torch.Tensor:
    return torch.ops.extension_cpp.mysoftmax.default(a.contiguous())

@torch.library.register_fake("extension_cpp::mysoftmax")
def _(a:torch.Tensor):
    torch._check(a.dtype == torch.float)
    return torch.empty_like(a)
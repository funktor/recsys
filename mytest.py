import torch
import extension_cpp
import time

a = torch.randn(10000, 1024, dtype=torch.float32)
b = a.to(device='cuda:0')

start = time.time()*1000
b1 = torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32)
end = time.time()*1000
print(end-start)

print(b1)

start = time.time()*1000
b2 = extension_cpp.mysoftmax_gpu(b.contiguous())
end = time.time()*1000
print(end-start)

print(b2)


class MySoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        output:torch.Tensor = extension_cpp.mysoftmax_cpu(input.contiguous())
        ctx.save_for_backward(output.contiguous())
        return output

    @staticmethod
    def backward(ctx, grad:torch.Tensor):
        output = extension_cpp.mysoftmax_cpu_grad(grad.contiguous(), *ctx.saved_tensors)
        return output
    
class MySoftmax(torch.nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, input):
        return MySoftmaxFunction.apply(input)
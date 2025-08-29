import torch
import extension_cpp
import time

class MySoftmaxFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        output:torch.Tensor = extension_cpp.mysoftmax_cpu(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad:torch.Tensor):
        output = extension_cpp.mysoftmax_cpu_grad(grad.contiguous(), *ctx.saved_tensors)
        return output
    
class MySoftmaxCPU(torch.nn.Module):
    def __init__(self):
        super(MySoftmaxCPU, self).__init__()

    def forward(self, input):
        return MySoftmaxFunctionCPU.apply(input)
    

class MySoftmaxFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        output:torch.Tensor = extension_cpp.mysoftmax_gpu(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad:torch.Tensor):
        output = extension_cpp.mysoftmax_gpu_grad(grad.contiguous(), *ctx.saved_tensors)    
        return output
    
class MySoftmaxGPU(torch.nn.Module):
    def __init__(self):
        super(MySoftmaxGPU, self).__init__()

    def forward(self, input):
        return MySoftmaxFunctionGPU.apply(input)
    

a_cpu = torch.randn(1000, 1024, dtype=torch.float32, requires_grad=True, device='cpu')

start = time.time()*1000
b1 = torch.nn.functional.softmax(a_cpu, dim=-1, dtype=torch.float32)
end = time.time()*1000
print("Torch CPU Forward Pass Duration = ", end-start)
print("Torch CPU Forward Pass Output\n", b1)
print()

start = time.time()*1000
(b1**2).sum().backward()
end = time.time()*1000
print("Torch CPU Backward Pass Duration = ", end-start)
print("Torch CPU Backward Pass Output\n", a_cpu.grad)
print()

b_cpu = a_cpu.clone()
b_cpu.retain_grad()

start = time.time()*1000
h = MySoftmaxCPU()
b2 = h(b_cpu)
end = time.time()*1000
print("Custom CPU Forward Pass Duration = ", end-start)
print("Custom CPU Forward Pass Output\n", b2)
print()

start = time.time()*1000
(b2**2).sum().backward()
end = time.time()*1000
print("Custom CPU Backward Pass Duration = ", end-start)
print("Custom CPU Backward Pass Output\n", b_cpu.grad)
print()

a_gpu = a_cpu.to(device='cuda:0')
a_gpu.retain_grad()

start = time.time()*1000
h = MySoftmaxGPU()
b3 = h(a_gpu)
end = time.time()*1000
print("Custom GPU Forward Pass Duration = ", end-start)
print("Custom GPU Forward Pass Output\n", b3)
print()

start = time.time()*1000
(b3**2).sum().backward()
end = time.time()*1000
print("Custom GPU Backward Pass Duration = ", end-start)
print("Custom GPU Backward Pass Output\n", a_gpu.grad)
print()
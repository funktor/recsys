import torch
import extension_cpp
import time

class MySoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        output:torch.Tensor = extension_cpp.mysoftmax_cpu(input) if input.device == 'cpu' \
            else extension_cpp.mysoftmax_gpu(input)
        ctx.save_for_backward(output.to(device=input.device))
        return output

    @staticmethod
    def backward(ctx, grad:torch.Tensor):
        output = \
            extension_cpp.mysoftmax_cpu_grad(grad.contiguous(), *ctx.saved_tensors) if grad.device == 'cpu' \
                else extension_cpp.mysoftmax_gpu_grad(grad.contiguous(), *ctx.saved_tensors)
        
        return output
    
class MySoftmax(torch.nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, input):
        return MySoftmaxFunction.apply(input)
    

a_cpu = torch.randn(5, 10, dtype=torch.float32, requires_grad=True, device='cpu')
a_gpu = a_cpu.to(device='cuda:0')

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

b_cpu = torch.Tensor(a_cpu, dtype=torch.float32, requires_grad=True, device='cpu')

start = time.time()*1000
h = MySoftmax()
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


start = time.time()*1000
h = MySoftmax()
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





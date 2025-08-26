import torch
import extension_cpp
import time

a = torch.randn(1000, 100, dtype=torch.float32, device='cuda:0')

start = time.time()*1000
b1 = torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32)
end = time.time()*1000
print(end-start)


start = time.time()*1000
b2 = extension_cpp.mysoftmax_gpu(a.contiguous())
end = time.time()*1000
print(end-start)


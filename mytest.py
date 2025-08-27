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

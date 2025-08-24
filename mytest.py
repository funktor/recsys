import torch
import extension_cpp
import time

a = torch.randn(10000, 1000, dtype=torch.float32)

start = time.time()*1000
b1 = torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32)
end = time.time()*1000
print(end-start)

start = time.time()*1000
b2 = extension_cpp.mysoftmax(a.contiguous())
end = time.time()*1000
print(end-start)

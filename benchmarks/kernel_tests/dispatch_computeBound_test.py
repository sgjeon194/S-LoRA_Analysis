import torch
from slora._kernels import dispatch_bgmv, dispatch_computeBound, dispatch_memoryBound
import numpy

data_length = 65536
copy_time = 64
result = torch.randn(data_length, dtype=torch.float, device="cuda")
A = torch.randn(data_length, dtype=torch.float, device="cuda")
B = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")
C = torch.randn(data_length, dtype=torch.float, device="cuda")
D = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")
E = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")

cycles = torch.zeros(65536, device='cuda', dtype=torch.long)

for i in range(5):
    torch.cuda.nvtx.range_push("Base")
    dispatch_computeBound(result)
    dispatch_memoryBound(B, A, 65536)
    torch.cuda.nvtx.range_pop()
    
# cycles_array = cycles.cpu().numpy()
# print(numpy.average(cycles_array))
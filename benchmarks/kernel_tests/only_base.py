import torch
import time


device = torch.device("cuda")
data_type = torch.float16
# input_size = 1200 # prefill
input_size = 1 # decode
batch_size = 64
rank = 8

X = torch.randn(batch_size * input_size, 4096, dtype=torch.float16, device=device)
W = torch.randn(4096, 4096, dtype=torch.float16, device=device)

for i in range(5):    
    torch.cuda.nvtx.range_push("Base")
    base_result = torch.mm(X, W) # (inputsize, 4096) * (4096, 4096)
    torch.cuda.nvtx.range_pop()
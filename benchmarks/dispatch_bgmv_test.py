import torch
from slora._kernels import dispatch_bgmv, stream_pass_test
import time
import nvtx

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

layer_num = 5

# input_size = 1200 # prefill
input_size = 1 # decode
batch_size = 1
rank = 8

W_group = []
X_group = []
A_group = []
B_group = []

a_start         = []
a_len           = []
a_loc           = []
batch_req_bins  = []
a_scaling       = []

base_result = []
lora_result = []

for i in range(layer_num):
    W_group.append(torch.randn(4096, 4096, device=device, dtype=torch.float16))
    X_group.append(torch.randn(batch_size * input_size, 4096, device=device, dtype=torch.float16))

    delta_qA = torch.zeros(batch_size * input_size, rank, device=device, dtype=torch.float16)
    A_group.append(torch.randn(10000, 32, 128, device=device, dtype=torch.float16))
    B_group.append(torch.randn(10000, 32, 128, device=device, dtype=torch.float16))

    a_start.append(         torch.tensor([0 for _ in range(batch_size)],        device=device, dtype=torch.long))
    a_len.append(           torch.tensor([rank * 4 for _ in range(batch_size)], device=device, dtype=torch.long))
    a_loc.append(           torch.tensor(range(rank * 4 * batch_size),          device=device, dtype=torch.long))
    batch_req_bins.append(  torch.zeros((batch_size * input_size,),    device=device, dtype=torch.long))
    a_scaling.append(       torch.tensor([0.5000 for _ in range(batch_size)],   device=device, dtype=torch.float16))

start = time.time()
for i in range(layer_num):
    torch.cuda.nvtx.range_push("Base")
    base_result.append(torch.mm(X_group[i], W_group[i])) # (inputsize, 4096) * (4096, 4096)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Shrink")
    dispatch_bgmv(delta_qA, X_group[i], A_group[i], a_start[i], a_len[i], a_loc[i], batch_req_bins[i], 0, a_scaling[i])
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Expand")
    dispatch_bgmv(base_result[i], delta_qA, B_group[i], a_start[i], a_len[i], a_loc[i], batch_req_bins[i], 0, a_scaling[i])
    torch.cuda.nvtx.range_pop()
    # (inputsize, 4096) * (4096, max_rank)

torch.cuda.synchronize()

print(f"Time : {time.time() - start}")
import torch
from slora._kernels import dispatch_bgmv, stream_pass_test
import time
import nvtx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_num = 5

batch_size = 4
prompt_size = 1200
rank = 32

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
    X_group.append(torch.randn(batch_size * prompt_size, 4096, device=device, dtype=torch.float16))

    delta_qA = torch.zeros(batch_size * prompt_size, 32, device=device, dtype=torch.float16)
    A_group.append(torch.randn(10000, 32, 128, device=device, dtype=torch.float16))
    B_group.append(torch.randn(10000, 32, 128, device=device, dtype=torch.float16))

    a_start.append(         torch.tensor([0],                           device=device, dtype=torch.long))
    a_len.append(           torch.tensor([rank * 4],                    device=device, dtype=torch.long))
    a_loc.append(           torch.tensor(range(rank * 4),               device=device, dtype=torch.long))
    batch_req_bins.append(  torch.zeros((batch_size * prompt_size,),    device=device, dtype=torch.long))
    a_scaling.append(       torch.tensor([0.5000],                      device=device, dtype=torch.float16))

torch.cuda.nvtx.mark("start")

start = time.time()
for i in range(layer_num):
    with nvtx.annotate(f"Layer {i}"):
        base_result.append(torch.mm(X_group[i], W_group[i])) # (inputsize, 4096) * (4096, 4096)
        dispatch_bgmv(delta_qA, X_group[i], A_group[i], a_start[i], a_len[i], a_loc[i], batch_req_bins[i], 0, a_scaling[i])
        lora_result.append(dispatch_bgmv(base_result[i], delta_qA, B_group[i], a_start[i], a_len[i], a_loc[i], batch_req_bins[i], 0, a_scaling[i]))
        # (inputsize, 4096) * (4096, max_rank)

print(f"Time : {time.time() - start}")
import torch
from slora._kernels import dispatch_bgmv, stream_pass_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

layer_num = 5

input_size = 4098
rank = 32

X = []
W = []
A = []
a_start         = []
a_len           = []
a_loc           = []
batch_req_bins  = []
a_scaling       = []

base_result = []
lora_result = []

for i in range(layer_num):
    X.append(torch.randn(input_size, 4096, device=device, dtype=torch.float16))
    W.append(torch.randn(4096, 4096, device=device, dtype=torch.float16))

    delta_qA = torch.zeros(input_size, 32, device=device, dtype=torch.float16)
    A.append(torch.randn(10000, 32, 128, device=device, dtype=torch.float16))

    a_start.append(             torch.tensor([0],           device=device, dtype=torch.long))
    a_len.append(torch.tensor(  [rank * 4],                 device=device, dtype=torch.long))
    a_loc.append(torch.tensor(  range(rank * 4),            device=device, dtype=torch.long))
    batch_req_bins.append(      torch.zeros((input_size,),  device=device, dtype=torch.long))
    a_scaling.append(           torch.tensor([0.5000],      device=device, dtype=torch.float16))

torch.cuda.nvtx.mark("start")

for i in range(layer_num):
    with torch.cuda.stream(stream1): 
        # (inputsize, 4096) * (4096, 4096)
        base_result.append(torch.mm(X[i], W[i]))
    with torch.cuda.stream(stream2):
        # (inputsize, 4096) * (4096, max_rank)
        lora_result.append(dispatch_bgmv(delta_qA, X[i], A[i], a_start[i], a_len[i], a_loc[i], batch_req_bins[i], 0, a_scaling[i], stream2.cuda_stream))

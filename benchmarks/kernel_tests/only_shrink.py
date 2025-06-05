import torch
from slora._kernels import dispatch_bgmv
import time


device = torch.device("cuda")
data_type = torch.float16
# input_size = 1200 # prefill
input_size = 1 # decode
batch_size = 1
rank = 8

use_multistream = True
use_cudagraph = True

print(f"batch_size {batch_size}")
X = torch.randn(batch_size * input_size, 4096, dtype=torch.float16, device=device)
W = torch.randn(4096, 4096, dtype=torch.float16, device=device)
A_batched = torch.zeros(10000, 32, 128, dtype=torch.float16, device=device)
B_batched = torch.zeros(10000, 32, 128, dtype=torch.float16, device=device)

A_list = []
for i in range(batch_size):
    q_lora_A = ((torch.rand((rank, 4096), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    k_lora_A = ((torch.rand((rank, 4096), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    v_lora_A = ((torch.rand((rank, 4096), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    o_lora_A = ((torch.rand((rank, 4096), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    q_lora_B = ((torch.rand((4096, rank), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    k_lora_B = ((torch.rand((4096, rank), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    v_lora_B = ((torch.rand((4096, rank), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)
    o_lora_B = ((torch.rand((4096, rank), dtype=data_type, device="cuda").transpose(0, 1).contiguous() * 2 - 1) * 1e-3)

    A = torch.concat(
        [q_lora_A.T.reshape(rank, 32, -1),
        k_lora_A.T.reshape(rank, 32, -1),
        v_lora_A.T.reshape(rank, 32, -1),
        o_lora_A.T.reshape(rank, 32, -1)])

    B = torch.concat(
        [q_lora_B.T.reshape(rank, 32, -1),
        k_lora_B.T.reshape(rank, 32, -1),
        v_lora_B.T.reshape(rank, 32, -1),
        o_lora_B.T.reshape(rank, 32, -1)])
    
    A_batched[4 * rank * i:4 * rank * (i + 1)] = A
    B_batched[4 * rank * i:4 * rank * (i + 1)] = B
    
    A_list.append(q_lora_A)

a_start = torch.tensor([rank * 4 * i for i in range(batch_size)],        device=device, dtype=torch.long)
a_len =   torch.tensor([rank * 4 for _ in range(batch_size)], device=device, dtype=torch.long)
a_loc =   torch.tensor(range(rank * 4 * batch_size),          device=device, dtype=torch.long)
batch_req_bin = [i for i in range(batch_size) for _ in range(input_size)]
batch_req_bin = torch.tensor(batch_req_bin, device=device, dtype=torch.long)
a_scaling = torch.tensor([0.5000 for _ in range(batch_size)],   device=device, dtype=torch.float16)

lora_stream = torch.cuda.Stream()
lora_stream_id = lora_stream.cuda_stream if use_multistream else torch.cuda.default_stream().cuda_stream

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

# for i in range(5):
shrink_result = torch.zeros(batch_size * input_size, rank, dtype=torch.float16, device=device)

torch.cuda.nvtx.range_push("Shrink")
dispatch_bgmv(shrink_result, X, A_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()


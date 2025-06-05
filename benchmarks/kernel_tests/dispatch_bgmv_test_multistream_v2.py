import torch
from slora._kernels import dispatch_bgmv
import time

python_start = time.time()
correction_test = False

device = torch.device("cuda")
data_type = torch.float16
# input_size = 1200 # prefill
input_size = 1 # decode
batch_size = 32
rank = 8

use_multistream = True
use_cudagraph = True

print(f"batch_size {batch_size}")
print(f"use_multistream {use_multistream}")
print(f"use_cudagraph {use_cudagraph}")

X = torch.randn(batch_size * input_size, 4096, dtype=torch.float16, device=device)
W = torch.randn(4096, 4096, dtype=torch.float16, device=device)
A_batched = torch.zeros(10000, 32, 128, dtype=torch.float16, device=device)
B_batched = torch.zeros(10000, 32, 128, dtype=torch.float16, device=device)

divided_batch_size = 8
punica_call_num = int(batch_size / divided_batch_size)

A_list = []
A_batched_list = []
B_batched_list = []
for punica_call in range(punica_call_num):
    for i in range(divided_batch_size):
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
    
    A_batched_list.append(A_batched)
    B_batched_list.append(B_batched)

a_start = torch.tensor([rank * 4 * i for i in range(divided_batch_size)],        device=device, dtype=torch.long)
a_len =   torch.tensor([rank * 4 for _ in range(divided_batch_size)], device=device, dtype=torch.long)
a_loc =   torch.tensor(range(rank * 4 * divided_batch_size),          device=device, dtype=torch.long)
batch_req_bin = [i for i in range(divided_batch_size) for _ in range(input_size)]
batch_req_bin = torch.tensor(batch_req_bin, device=device, dtype=torch.long)
a_scaling = torch.tensor([0.5000 for _ in range(divided_batch_size)],   device=device, dtype=torch.float16)

lora_stream = torch.cuda.Stream()
lora_stream_id = lora_stream.cuda_stream if use_multistream else torch.cuda.default_stream().cuda_stream

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

_ = torch.mm(X, W)

one_layer_graph = torch.cuda.CUDAGraph()
shrink_result = torch.zeros(divided_batch_size * input_size, rank, dtype=torch.float16, device=device)
expand_result = torch.zeros(divided_batch_size * input_size, 4096, dtype=torch.float16, device=device)

lora_result = torch.zeros(batch_size * input_size, 4096, dtype=torch.float16, device=device)

if use_cudagraph:
    with torch.cuda.graph(one_layer_graph):
        lora_stream.wait_stream(torch.cuda.current_stream())
        
        torch.cuda.nvtx.range_push("Base")
        base_result = torch.mm(X, W) # (inputsize, 4096) * (4096, 4096)
        torch.cuda.nvtx.range_pop()

        #torch.cuda.set_stream(lora_stream)
        with torch.cuda.stream(lora_stream):
            for j in range(punica_call_num):
                shrink_result = torch.zeros(divided_batch_size * input_size, rank, dtype=torch.float16, device=device)
                expand_result = torch.zeros(divided_batch_size * input_size, 4096, dtype=torch.float16, device=device)
                torch.cuda.nvtx.range_push("Shrink")
                dispatch_bgmv(shrink_result, X[j:j+divided_batch_size], A_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_push("Expand")
                dispatch_bgmv(expand_result, shrink_result, B_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, 4096)
                torch.cuda.nvtx.range_pop()

                lora_result[divided_batch_size * j:divided_batch_size * (j + 1)] = expand_result
        
        torch.cuda.current_stream().wait_stream(lora_stream)
        
        torch.cuda.nvtx.range_push("Add")
        result = base_result + lora_result
        torch.cuda.nvtx.range_pop()

    for i in range(5):
        torch.cuda.nvtx.range_push("Cudagraph")
        one_layer_graph.replay()
        torch.cuda.nvtx.range_pop()


else:
    for i in range(5):
        lora_stream.wait_stream(torch.cuda.current_stream())
        
        torch.cuda.nvtx.range_push("Base")
        base_result = torch.mm(X, W) # (inputsize, 4096) * (4096, 4096)
        torch.cuda.nvtx.range_pop()

        #torch.cuda.set_stream(lora_stream)
        with torch.cuda.stream(lora_stream):
            for j in range(punica_call_num):
                shrink_result = torch.zeros(divided_batch_size * input_size, rank, dtype=torch.float16, device=device)
                expand_result = torch.zeros(divided_batch_size * input_size, 4096, dtype=torch.float16, device=device)
                torch.cuda.nvtx.range_push("Shrink")
                dispatch_bgmv(shrink_result, X[j:j+divided_batch_size], A_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
                torch.cuda.nvtx.range_pop()
                
                torch.cuda.nvtx.range_push("Expand")
                dispatch_bgmv(expand_result, shrink_result, B_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, 4096)
                torch.cuda.nvtx.range_pop()

                lora_result[divided_batch_size * j:divided_batch_size * (j + 1)] = expand_result
        
        torch.cuda.current_stream().wait_stream(lora_stream)
        
        torch.cuda.nvtx.range_push("Add")
        result = base_result + lora_result
        torch.cuda.nvtx.range_pop()


python_end = time.time()
        
print(python_end - python_start)

if correction_test:
    shrink_result_test = torch.zeros_like(shrink_result, dtype=data_type, device='cuda')
    for i in range(batch_size):
        shrink_result_test[input_size * i:input_size * (i + 1)] = torch.mm(X[input_size * i:input_size * (i + 1)], A_list[i])
        
    for i in range(4096):
        issame = torch.allclose(shrink_result[i], shrink_result_test[i], 1e-3)
        if not issame:
            print(f"{i} {shrink_result[i]} {shrink_result_test[i]}")

torch.cuda.synchronize()

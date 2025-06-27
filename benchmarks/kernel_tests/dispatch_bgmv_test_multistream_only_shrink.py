import torch
from slora._kernels import dispatch_bgmv, dispatch_computeBound, dispatch_memoryBound
import time
import numpy

correction_test = False

device = torch.device("cuda")
data_type = torch.float16
# input_size = 1200 # prefill
input_size = 1 # decode
batch_size = 1
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


base_stream = torch.cuda.Stream()
lora_stream = torch.cuda.Stream()
lora_stream_id = lora_stream.cuda_stream if use_multistream else base_stream.cuda_stream

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

_ = torch.mm(X, W)

one_layer_graph = torch.cuda.CUDAGraph()
shrink_result = torch.zeros(batch_size * input_size, rank, dtype=torch.float16, device=device)
expand_result = torch.zeros(batch_size * input_size, 4096, dtype=torch.float16, device=device)

data_length = 65536
copy_time = 60
result = torch.randn(data_length, dtype=torch.float, device="cuda")
A = torch.randn(data_length, dtype=torch.float, device="cuda")
B = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")
C = torch.randn(data_length, dtype=torch.float, device="cuda")
D = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")
E = torch.zeros(data_length * copy_time, dtype=torch.float, device="cuda")

cycles = torch.zeros(65536, device='cuda', dtype=torch.long)

if use_cudagraph:
    
    with torch.cuda.graph(one_layer_graph):
        base_stream.wait_stream(torch.cuda.current_stream())
        lora_stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(base_stream):
            torch.cuda.nvtx.range_push("Base")
            base_result = torch.mm(X, W) # (inputsize, 4096) * (4096, 4096)
            # dispatch_computeBound(A, base_stream.cuda_stream) 
            # dispatch_memoryBound(B, A, 32768, base_stream.cuda_stream)
            torch.cuda.nvtx.range_pop()

        #torch.cuda.set_stream(lora_stream)
        # with torch.cuda.stream(lora_stream):
        #     torch.cuda.nvtx.range_push("Shrink")
        #     # dispatch_bgmv(shrink_result, X, A_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
        #     # dispatch_computeBound(result, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
        #     dispatch_memoryBound(D, C, data_length, lora_stream.cuda_stream)
        #     torch.cuda.nvtx.range_pop()
            
        torch.cuda.current_stream().wait_stream(base_stream)
        # torch.cuda.current_stream().wait_stream(lora_stream)
        # E = B + D
    
    python_start = time.time()
    for i in range(15):
        # E.zero_()
        # B.zero_()
        # D.zero_()
        # shrink_result = torch.zeros(batch_size * input_size, rank, dtype=torch.float16, device=device)
        # expand_result = torch.zeros(batch_size * input_size, 4096, dtype=torch.float16, device=device)
        one_layer_graph.replay()

        
    # cycles_array = cycles.cpu().numpy()
    # print(numpy.average(cycles_array))
    
    python_end = time.time()
else:
    python_start = time.time()
    
    for i in range(5):
        shrink_result = torch.zeros(batch_size * input_size, rank, dtype=torch.float16, device=device)
        expand_result = torch.zeros(batch_size * input_size, 4096, dtype=torch.float16, device=device)
        lora_stream.wait_stream(torch.cuda.current_stream())
        
        torch.cuda.nvtx.range_push("Base")
        base_result = torch.mm(X, W) # (inputsize, 4096) * (4096, 4096)
        # dispatch_computeBound(A) 
        # dispatch_memoryBound(B, A, 65536)
        torch.cuda.nvtx.range_pop()

        #torch.cuda.set_stream(lora_stream)
        with torch.cuda.stream(lora_stream):
            torch.cuda.nvtx.range_push("Shrink")
            # dispatch_bgmv(shrink_result, X, A_batched, a_start, a_len, a_loc, batch_req_bin, 0, a_scaling, lora_stream_id) # (inputsize, 4096) * (4096, max_rank)
            dispatch_computeBound(result, lora_stream_id) 
            # dispatch_memoryBound(D, C, 65536, lora_stream.cuda_stream)
            torch.cuda.nvtx.range_pop()
            
        torch.cuda.current_stream().wait_stream(lora_stream)
        
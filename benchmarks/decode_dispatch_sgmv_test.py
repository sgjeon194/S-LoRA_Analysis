import torch
from slora._kernels import dispatch_bgmv, stream_pass_test
import time
import nvtx
import random
import numpy as np
import argparse

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
data_type = torch.float16
    
qkvo = 0
loop = 10

def naive_lora(batched_input:torch.Tensor, A_list:list, B_list:list, using_lora_indices:list):
    assert len(A_list) == len(B_list)
    batch_size = len(using_lora_indices)
    assert batch_size != 0
    assert A_list[0].shape[0] == B_list[0].shape[1] 
    dmodel = A_list[0].shape[0]
    assert A_list[0].shape[1] == B_list[0].shape[0] 
    rank = int(A_list[0].shape[1] / 4)
    
    shrink_result = torch.zeros(batch_size, rank, device=device, dtype=data_type)
    expand_result = torch.zeros(batch_size, dmodel, device=device, dtype=data_type)
    static_A_list = []
    static_B_list = []
    for i in range(batch_size):
        static_A_list.append(torch.zeros(A_list[0].shape[0], int(A_list[0].shape[1] / 4), device='cuda', dtype=data_type))
        static_B_list.append(torch.zeros(int(B_list[0].shape[0] / 4), B_list[0].shape[1], device='cuda', dtype=data_type))    

    naive_cudagraph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    
    with torch.cuda.graph(naive_cudagraph):
        torch.cuda.nvtx.range_push("Naive lora")
        for i in range(batch_size):
            torch.matmul(batched_input[i:i+1], static_A_list[i], out=shrink_result[i:i+1])
            torch.matmul(shrink_result[i:i+1], static_B_list[i], out=expand_result[i:i+1])
        torch.cuda.nvtx.range_pop()
        
    for i in range(batch_size):
        static_A_list[i].copy_(A_list[using_lora_indices[i]][:, rank * qkvo:rank * (qkvo+1)])
        static_B_list[i].copy_(B_list[using_lora_indices[i]][rank * qkvo:rank * (qkvo+1), :])
        
    for _ in range(loop):
        torch.cuda.nvtx.range_push("1 loop - naive")
        naive_cudagraph.replay()
        torch.cuda.nvtx.range_pop()


    return expand_result

def bmm_lora(batched_input:torch.Tensor, A_list:list, B_list:list, using_lora_indices:list):
    assert len(A_list) == len(B_list)
    batch_size = len(using_lora_indices)
    assert batch_size != 0
    assert A_list[0].shape[0] == B_list[0].shape[1] 
    dmodel = A_list[0].shape[0]
    assert A_list[0].shape[1] == B_list[0].shape[0] 
    rank = int(A_list[0].shape[1] / 4)
    
    batched_input = batched_input.unsqueeze(1)

    batched_A = torch.zeros(batch_size, dmodel, rank, device='cuda', dtype=data_type)
    shrink_result = torch.zeros(batch_size, 1, rank, device='cuda', dtype=data_type)

    batched_B = torch.zeros(batch_size, rank, dmodel, device='cuda', dtype=data_type)
    expand_result = torch.zeros(batch_size, 1, dmodel, device='cuda', dtype=data_type)
    
    bmm_cudagraph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(bmm_cudagraph):
        
        torch.cuda.nvtx.range_push("Gather")
        for i in range(batch_size):
            batched_A[i].copy_(A_list[using_lora_indices[i]][:, rank * qkvo:rank * (qkvo+1)])
            batched_B[i].copy_(B_list[using_lora_indices[i]][rank * qkvo:rank * (qkvo+1), :])
        torch.cuda.nvtx.range_pop()
                        
        torch.cuda.nvtx.range_push("Bmm lora")
        torch.bmm(batched_input, batched_A, out=shrink_result)
        torch.bmm(shrink_result, batched_B, out=expand_result)
        torch.cuda.nvtx.range_pop()
    
    for _ in range(loop):
        torch.cuda.nvtx.range_push("1 loop - bmm")
        bmm_cudagraph.replay()
        torch.cuda.nvtx.range_pop()
    
    return expand_result

def bgmv_lora(batched_input:torch.Tensor, A_list:list, B_list:list, using_lora_indices:list):
    assert len(A_list) == len(B_list)
    batch_size = len(using_lora_indices)
    assert batch_size != 0
    assert A_list[0].shape[0] == B_list[0].shape[1] 
    dmodel = A_list[0].shape[0]
    assert A_list[0].shape[1] == B_list[0].shape[0] 
    rank = int(A_list[0].shape[1] / 4)
    
    unique_adapter_num = max(using_lora_indices) + 1
    
    # Lora meta data
    a_loc=torch.empty(0, dtype=torch.long, device="cuda")
    a_start=torch.empty(0, dtype=torch.long, device="cuda")
    a_len=torch.empty(0, dtype=torch.long, device="cuda")
    a_scaling=torch.empty(0, dtype=data_type, device="cuda")

    new_loc = torch.tensor(range(4 * rank * unique_adapter_num), dtype=torch.int64, device='cuda')
    start_offset = 0
    a_start = torch.cat((a_start, torch.empty(unique_adapter_num, dtype=torch.long, device="cuda")))
    len_offset = a_len.shape[0]
    a_len = torch.cat((a_len, torch.empty(unique_adapter_num, dtype=torch.long, device="cuda")))
    loc_offset = a_loc.shape[0]
    a_loc = torch.cat((a_loc, torch.empty(4 * rank * unique_adapter_num, dtype=torch.long, device="cuda")))

    req_bins = torch.tensor(using_lora_indices, device='cuda', dtype=torch.int64)

    cum_loc = 0
    cum_loc_list = []
    for i in range(unique_adapter_num):
        cum_loc_list.append(cum_loc)
        a_start[start_offset + i] = loc_offset + cum_loc
        a_len[len_offset + i] = rank * 4
        a_loc[loc_offset + cum_loc: loc_offset + cum_loc + rank * 4] = (new_loc[cum_loc: cum_loc + rank * 4])
        cum_loc += rank * 4
    a_scaling = torch.cat((a_scaling, torch.tensor([1.0 for _ in range(unique_adapter_num)], dtype=data_type, device="cuda")))


    # Lora data arrange
    key_buffer = torch.zeros(batch_size * 2048, 32, 128, device='cuda', dtype=data_type).contiguous()
    val_buffer = torch.zeros(10000, 32, 128, device='cuda', dtype=data_type).contiguous()

    shrink_result = torch.zeros(batch_size, rank, device=device, dtype=data_type)
    expand_result = torch.zeros(batch_size, dmodel, device=device, dtype=data_type)

    # bgmv_cudagraph = torch.cuda.CUDAGraph()
    # torch.cuda.synchronize()
    # with torch.cuda.graph(bgmv_cudagraph):
    #     dispatch_bgmv(shrink_result, batched_input, key_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling)
    #     dispatch_bgmv(expand_result, shrink_result, val_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling)

    # Remake B for bgmv
    new_B_list = []
    for i in range(len(B_list)):
        b = B_list[i].transpose(0, 1)
        q = b[:, 0*rank:1*rank].reshape(rank, 32, -1)
        k = b[:, 1*rank:2*rank].reshape(rank, 32, -1)
        v = b[:, 2*rank:3*rank].reshape(rank, 32, -1)
        o = b[:, 3*rank:4*rank].reshape(rank, 32, -1)
        new_B_list.append(torch.concat([q,k,v,o]))

    for i in range(unique_adapter_num):
        cum_loc = cum_loc_list[i]
        loc = new_loc[cum_loc: cum_loc + rank * 4]
        a = A_list[i].T.reshape(4 * rank, 32, 128).contiguous()
        b = new_B_list[i].contiguous()
        key_buffer.index_copy_(0, loc, a)
        val_buffer.index_copy_(0, loc, b)

    # for _ in range(loop):
    #     shrink_result.zero_()
    #     expand_result.zero_()
    #     torch.cuda.nvtx.range_push("1 loop - bgmv slora")
    #     dispatch_bgmv(shrink_result, batched_input, key_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling)
    #     dispatch_bgmv(expand_result, shrink_result, val_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling)
    #     torch.cuda.nvtx.range_pop()


    bgmv_cudagraph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(bgmv_cudagraph):
        dispatch_bgmv(shrink_result, batched_input, key_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling, torch.cuda.current_stream().cuda_stream)
        dispatch_bgmv(expand_result, shrink_result, val_buffer, a_start, a_len, a_loc, req_bins, qkvo, a_scaling, torch.cuda.current_stream().cuda_stream)


    for _ in range(loop):
        # shrink_result.zero_()
        # expand_result.zero_()
        torch.cuda.nvtx.range_push("1 loop - bgmv slora")
        bgmv_cudagraph.replay()
        torch.cuda.nvtx.range_pop()
        

    return expand_result

def make_distinct_list(n:int):
    arr = np.arange(n)
    np.random.shuffle(arr)
    return arr.tolist()

def make_uniform_list(n:int):
    k = int(n**0.5)
    arr = np.repeat(np.arange(k), int(n/k))
    arr = np.concatenate((arr, np.arange(int(n%k))))
    np.random.shuffle(arr)
    return arr.tolist()

def make_identical_list(n:int):
    arr = [0] * n
    return arr

def main():
    parser = argparse.ArgumentParser()

    # 인자 추가
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of decode")
    parser.add_argument("--lin", type=int, default=1, help="prompt length of each request")
    parser.add_argument("--distribution", type=int, default=2, help="Enter the lora distribution type. (Distinct : 1, Uniform : 2, Identical : 3)")

    args = parser.parse_args()
    
    batch_size = args.batch_size
    prompt_len = args.lin
    distribution = args.distribution
    
    dmodel = 4096
    rank = 16
    
    if distribution == 1:
        print("Distinct")
        using_lora_ids = make_distinct_list(batch_size)
    elif distribution == 2:
        print("Uniform")
        using_lora_ids = make_uniform_list(batch_size)
    elif distribution == 3:
        print("Identical")
        using_lora_ids = make_identical_list(batch_size)
    else:
        print("Check the lora distribution type (Random)")
        using_lora_ids = random.choices(range(unique_adapter_num), k=batch_size)
    
    using_lora_ids = sorted(using_lora_ids)
    unique_adapter_num = max(using_lora_ids) + 1
    
    
    # Input data
    X = torch.randn(batch_size, dmodel, device='cuda', dtype=data_type)

    # Model data
    W = torch.randn(dmodel, dmodel, device='cuda', dtype=data_type)

    # Lora data
    A_list = []
    B_list = []
    for i in range(unique_adapter_num):
        A = torch.randn(dmodel, 4 * rank, device='cuda', dtype=data_type)
        B = torch.randn(4 * rank, dmodel, device='cuda', dtype=data_type)
        A_list.append(A)
        B_list.append(B)
    
    
    
    # Real run ----------------------------------------------------------------------------------------------------
    torch.cuda.nvtx.range_push("Base")
    for _ in range(loop):
        O_base = torch.nn.functional.linear(X, W) # shape [batch_size, dmodel]
    torch.cuda.nvtx.range_pop()
    print("\nBase")
    print(O_base.shape)
    
    naive_result = naive_lora(X, A_list, B_list, using_lora_ids)
    bmm_result = bmm_lora(X, A_list, B_list, using_lora_ids)
    bgmv_result = bgmv_lora(X, A_list, B_list, using_lora_ids)
        
    print(naive_result)
    print(bmm_result)
    print(bgmv_result)




if __name__ == "__main__":
    main()



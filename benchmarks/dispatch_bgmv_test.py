import torch
import time
from slora._kernels import dispatch_bgmv, stream_pass_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

def measure_matmul():
    X1 = torch.randn(1024, 4096, device=device, dtype=torch.float16)
    W1 = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    
    delta_qA1 = torch.zeros(1024, 32, device=device, dtype=torch.float16)
    A1 = torch.randn(10000, 32, 128, device=device, dtype=torch.float16)
    
    a_start1         = torch.tensor([0],         device=device, dtype=torch.long)
    a_len1           = torch.tensor([128],       device=device, dtype=torch.long)
    a_loc1           = torch.tensor(range(128),  device=device, dtype=torch.long)
    batch_req_bins1  = torch.zeros((1024,),      device=device, dtype=torch.long)
    a_scaling1       = torch.tensor([0.5000],    device=device, dtype=torch.float16)


    # X2 = torch.randn(1024, 4096, device=device, dtype=torch.float16)
    # W2 = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    # delta_qA2 = torch.zeros(1024, 32, device=device, dtype=torch.float16)
    # A2 = torch.randn(10000, 32, 128, device=device, dtype=torch.float16)
    
    # a_start2         = torch.tensor([0],         device=device, dtype=torch.long)
    # a_len2           = torch.tensor([128],       device=device, dtype=torch.long)
    # a_loc2           = torch.tensor(range(128),  device=device, dtype=torch.long)
    # batch_req_bins2  = torch.zeros((1024,),      device=device, dtype=torch.long)
    # a_scaling2       = torch.tensor([0.5000],    device=device, dtype=torch.float16)

    torch.cuda.nvtx.mark("start")
    
    with torch.cuda.stream(stream1): 
        torch.mm(X1, W1)
    with torch.cuda.stream(stream2):
        dispatch_bgmv(delta_qA1, X1, A1, a_start1, a_len1, a_loc1, batch_req_bins1, 0, a_scaling1, stream1.cuda_stream) # (inputsize, 4096) * (4096, max_rank)
    
    # time.sleep(5)
    
    # with torch.cuda.stream(stream1): 
    #     dispatch_bgmv(delta_qA1, X1, A1, a_start1, a_len1, a_loc1, batch_req_bins1, 0, a_scaling1, stream1.cuda_stream) # (inputsize, 4096) * (4096, max_rank)
    # with torch.cuda.stream(stream2):
    #     dispatch_bgmv(delta_qA2, X2, A2, a_start2, a_len2, a_loc2, batch_req_bins2, 0, a_scaling2, stream2.cuda_stream) # (inputsize, 4096) * (4096, max_rank)

    return

measure_matmul()
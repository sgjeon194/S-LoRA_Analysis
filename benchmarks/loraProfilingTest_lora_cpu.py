import torch
from slora._kernels import dispatch_bgmv, stream_pass_test
import time
import nvtx

device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

layer_num = 5

batch_size = 4
prompt_size = 1200
rank = 32

W_group = []
X_group = []
X_group_host = []
A_group = []
B_group = []

results = []
for i in range(layer_num):
    W_group.append(torch.randn(4096, 4096, device=device_cuda, dtype=torch.float16))
    X_group_host.append(torch.randn(batch_size * prompt_size, 4096, device=device_cpu, dtype=torch.float32))
    X_group.append(X_group_host[i].to(device=device_cuda, dtype=torch.float16))

    A_batch_group = []
    B_batch_group = []

    for j in range(batch_size):
        A_batch_group.append(torch.randn(4096, 32, device=device_cpu, dtype=torch.float32))
        B_batch_group.append(torch.randn(32, 4096, device=device_cpu, dtype=torch.float32))
        
    A_group.append(A_batch_group)
    B_group.append(B_batch_group)
    lora_result_host = torch.zeros_like(X_group_host[0]).pin_memory()
    lora_result = torch.zeros_like(X_group[0])


start = time.time()
for i in range(layer_num):
    with nvtx.annotate(f"Layer {i}"):
        base_result = torch.mm(X_group[i], W_group[i]) # (inputsize, 4096) * (4096, 4096)
        for j in range(batch_size):
            with nvtx.annotate(f"Lora {j}"):
                shrink_result = torch.mm(X_group_host[i][1200 * j:1200 * (j + 1), :], A_group[i][j])
                lora_result_host[1200 * j:1200 * (j + 1), :] = torch.mm(shrink_result, B_group[i][j])
            lora_result[1200 * j:1200 * (j + 1), :] = lora_result_host[1200 * j:1200 * (j + 1), :].to(device=device_cuda, dtype=torch.float16, non_blocking=True)
        
        torch.cuda.synchronize()
        result = base_result + lora_result

print(f"Time : {time.time() - start}")
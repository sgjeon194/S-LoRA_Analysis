import pandas as pd
import numpy as np
from benchmarks.trace import Request
import os

def dummy_prompt(prompt_len):
    return "Hello " * prompt_len

def generate_requests(num_adapters, alpha, adapter_dirs, trace_type, seed=42):
    np.random.seed(seed)
    
    TRACE_NAMES = [
        "Coding",
        "Conversation",
    ]

    TRACE_FILENAMES = [
        "AzureLLMInferenceTrace_code.csv",
        "AzureLLMInferenceTrace_conv.csv",
    ]

    # Read all traces
    df_traces = {}

    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/" + trace_filename, parse_dates=["TIMESTAMP"])
        
    trace = df_traces[TRACE_NAMES[trace_type]]
    
    # generate adapter id
    tot_req = len(trace)
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    requests = []
    tic = 0
    base_time = trace["TIMESTAMP"][0]
    for i, row in trace.iterrows():
        tic = row["TIMESTAMP"] - base_time
        input_len = row["ContextTokens"]
        #input_len = min(input_len, 2048)
        output_len = row["GeneratedTokens"]
        
        request = Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), input_len, output_len, tic.total_seconds())
        
        requests.append(request)
        
    return requests

def generate_downsampled_requests(num_adapters, alpha, req_rate, duration, input_range, output_range, adapter_dirs, trace_type, seed=42):
    np.random.seed(seed)
    
    TRACE_NAMES = [
        "Coding",
        "Conversation",
    ]

    TRACE_FILENAMES = [
        "AzureLLMInferenceTrace_code.csv",
        "AzureLLMInferenceTrace_conv.csv",
    ]

    # Read all traces
    df_traces = {}

    for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
        df_traces[trace_name] = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/" + trace_filename, parse_dates=["TIMESTAMP"])
        
    trace = df_traces[TRACE_NAMES[trace_type]]
    trace = downsample(trace, req_rate, duration, input_range, output_range)
    
    # generate adapter id
    total_request_num = len(trace)
    probs = np.random.power(alpha, total_request_num)
    adapter_id = (probs * num_adapters).astype(int)

    trace = rescale_request_time(trace, duration)

    requests = []
    for i, row in trace.iterrows():
        tic = row["TIMESTAMP"]
        input_len = int(row["ContextTokens"])
        output_len = int(row["GeneratedTokens"])
        
        request = Request(i, adapter_dirs[adapter_id[i]][0], adapter_dirs[adapter_id[i]][1],
                                dummy_prompt(input_len), input_len, output_len, tic)
        
        requests.append(request)
        
    return requests
    

def downsample(trace, req_rate, duration, input_range, output_range):
    request_num = int(req_rate * duration)
    select_ratio = 16
    selected_indicies = np.random.choice(len(trace), request_num * select_ratio, replace=False)
    selected_indicies.sort()
    # downsampled_trace = [trace.iloc[idx] for idx in selected_indicies]
    downsampled_trace = trace.iloc[selected_indicies]
    
    drop_ind = []
    for idx, row in downsampled_trace.iterrows():
        prompt_len = row["ContextTokens"]
        output_len = row["GeneratedTokens"]
        
        if prompt_len < input_range[0] or output_len < output_range[0] or \
            prompt_len > input_range[1] or output_len > output_range[1]:
            drop_ind.append(idx)

    downsampled_trace = downsampled_trace.drop(drop_ind).reset_index(drop=True)
    downsampled_trace = downsampled_trace[:request_num]
    print(f"Downsampled {len(downsampled_trace)}")
    
    return downsampled_trace

def rescale_request_time(trace, duration):
    interval_start = trace["TIMESTAMP"][0]
    interval_end = trace["TIMESTAMP"][len(trace) - 1]
    
    interval = interval_end - interval_start
    trace["TIMESTAMP"] = (trace["TIMESTAMP"] - interval_start) / interval * duration
    
    return trace
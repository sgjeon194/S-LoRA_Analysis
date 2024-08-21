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
        
    trace = df_traces["Conversation"]
    
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
        input_len = min(input_len, 2048)
        output_len = row["GeneratedTokens"]
        
        request = Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), input_len, output_len, tic.total_seconds())
        
        requests.append(request)
        
    return requests
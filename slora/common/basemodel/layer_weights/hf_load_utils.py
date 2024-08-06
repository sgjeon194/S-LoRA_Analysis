import gc
import os
from safetensors import safe_open
import torch
from tqdm import tqdm


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None,
                    dummy=False):
    data_type = torch.float16 if data_type == 'fp16' else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    
    if dummy:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(None, dummy=dummy)
        if transformer_layer_list is not None:
            model_name = weight_dir.rstrip("/").split("/")[-1]
            for layer in tqdm(transformer_layer_list, desc=f"load model {model_name}"):
                layer.load_hf_weights(None, dummy=dummy)
        return

    use_safetensors = True
    files = os.listdir(weight_dir)
    print(f"<< Weight dir path : {os.path.abspath(weight_dir)}>>")
    print(f"\tInside there are : {files}")
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    print(f"\tused files : {candidate_files} len : {len(candidate_files)}")
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
        print(f"\tnew used files : {candidate_files} len : {len(candidate_files)}")
    
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    for file_ in candidate_files:
        print(f"\tLoading weight : {file_}")
        if use_safetensors:
            weights = safe_open(os.path.join(weight_dir, file_), 'pt', 'cpu')
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
        else:
            weights = torch.load(os.path.join(weight_dir, file_), 'cpu')
            
        print(f"\tWeight's type : {type(weights)}")
        print(f"\tPre Post layer : {type(pre_post_layer)}")
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weights)
            
        print(f"\tTransformer layer : {type(transformer_layer_list[0])} / len : {len(transformer_layer_list)}")
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weights)
        del weights
        gc.collect()
    return

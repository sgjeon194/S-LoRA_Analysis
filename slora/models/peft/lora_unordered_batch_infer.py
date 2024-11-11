import numpy as np
import torch
import torch.nn as nn
from typing import final

from slora.common.infer_utils import init_bloc
from slora.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from slora.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from slora.models.peft.triton_kernel.lora.lora_prefill import lora_get_qkvo_fwd_shrink, lora_get_qkvo_fwd_expand
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.utils.infer_utils import mark_cost_time
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora._kernels import dispatch_bgmv, stream_pass_test

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time

class LoraUnorderedBatchInfer:

    def __init__(self, base_model, adapters, use_sync=False, infer_adapter=None):
        self.base_model = base_model

        lora_layer_dim = [adapter.r if adapter is not None else 0 for adapter in adapters]
        self.max_lora_dim = max(lora_layer_dim)

        self.req_bins = torch.zeros(len(adapters), dtype=torch.long, device="cuda")

        if infer_adapter is not None:
            self.infer_adapter = infer_adapter
            if isinstance(infer_adapter, NaiveInferAdapter):
                self.key_buffer = infer_adapter.key_buffer
                self.value_buffer = infer_adapter.value_buffer
            else:
                self.key_buffer = infer_adapter.mem_manager.key_buffer
                self.value_buffer = infer_adapter.mem_manager.value_buffer
            for i, adapter in enumerate(adapters):
                # FIX ME @TODO: currently not supporting adapter is None
                if adapter is None: continue
                idx = infer_adapter.adapter_dirs.index(adapter.lora_dir)
                self.req_bins[i] = idx
        
        self.kv_embed_dim = base_model.tp_k_head_num_ * base_model.head_dim_
        self.use_sync = use_sync
        self.timeDict = {}
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
    
    @torch.no_grad()
    def forward(
            self,
            batch_size, # number of request
            total_token_num,
            max_len_in_batch,
            input_ids, # 1D input tensor
            b_loc, # mapping to memory pool
            b_start_loc, # the start index of each request
            b_seq_len, # the current length of each request
            is_prefill=True,
            use_bmm=True,
            no_lora_compute=False,
            no_lora_copy=False):

        # Notice that batch_lora only support decoding
        assert len(b_loc) == len(b_start_loc) == len(b_seq_len)
        self.delta = []

        self.max_b_seq_len = torch.max(b_seq_len).item()

        if is_prefill:
            self.timeDict["arithType"] = "prefill"
            assert(len(self.req_bins)==len(b_seq_len))
            self.batch_req_bins = torch.repeat_interleave(self.req_bins, b_seq_len)
            # self.b_start_loc = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.long, device="cuda"), b_seq_len[:-1]]), dim=0)
            for _ in range(3):
                self.delta.append(torch.zeros((len(self.batch_req_bins), self.max_lora_dim), dtype=torch.float16, device="cuda"))

            return self._prefill(batch_size, total_token_num, max_len_in_batch,
                                 input_ids,
                                 b_loc, b_start_loc, b_seq_len, no_lora_compute)
        else:
            self.timeDict["arithType"] = "decode"
            for _ in range(3):
                self.delta.append(torch.zeros((len(b_seq_len), self.max_lora_dim), dtype=torch.float16, device="cuda"))

            return self._decode(batch_size, total_token_num, max_len_in_batch,
                                input_ids,
                                b_loc, b_start_loc, b_seq_len,
                                no_lora_compute, no_lora_copy)


    def _prefill(self, batch_size, total_token_num, max_len_in_batch,
                 input_ids,
                 b_loc, b_start_loc, b_seq_len, no_lora_compute=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (input_ids.shape[0] == total_token_num)
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        b_seq_len_numpy = b_seq_len.cpu().numpy()
        position_ids = torch.from_numpy(np.concatenate([np.arange(0, b_seq_len_numpy[i])
                                        for i in range(len(b_seq_len_numpy))], axis=0)).cuda()
        infer_state.position_cos = torch.index_select(
                self.base_model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
        infer_state.position_sin = torch.index_select(
                self.base_model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        position_ids = None

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.mem_manager = self.base_model.mem_manager
        infer_state.prefill_mem_index = self.base_model.mem_manager.alloc(infer_state.total_token_num)
        infer_state.prefill_key_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        infer_state.prefill_value_buffer = torch.empty(
                (infer_state.total_token_num, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                dtype=torch.float16, device="cuda")
        init_bloc(b_loc, b_seq_len, max_len_in_batch, infer_state.prefill_mem_index)
        # print("\n\t<<Prefill>>")
        # print(f"\t\tbatch_size {batch_size}")
        
        prefill_start_time = time.time()
        predict_logics = self._context_forward(input_ids, infer_state, no_lora_compute)
        if self.use_sync:
            torch.cuda.synchronize()
        
        # print(f"\t<Prefill end> --- time : {(prefill_end_time - prefill_start_time):0.8} ms -------------")
        #self.timeDict["total_time"] = (prefill_end_time - prefill_start_time)
        self.timeDict["total_time"] = 1000 * (time.time() - prefill_start_time)

        return predict_logics


    def _decode(self, batch_size, total_token_num, max_len_in_batch,
                input_ids,
                b_loc, b_start_loc, b_seq_len, no_lora_compute=False, no_lora_copy=False):
        infer_state = self.base_model.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

        infer_state.b_loc = b_loc
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        
        infer_state.mem_manager = self.base_model.mem_manager

        alloc_mem = self.base_model.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.decode_is_contiguous = True
            infer_state.decode_mem_index = alloc_mem[0]
            infer_state.decode_mem_start = alloc_mem[1]
            infer_state.decode_mem_end = alloc_mem[2]
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
        else:
            infer_state.decode_is_contiguous = False
            alloc_mem = self.base_model.mem_manager.alloc(batch_size)
            infer_state.decode_mem_index = alloc_mem
            infer_state.decode_key_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            infer_state.decode_value_buffer = torch.empty(
                    (batch_size, self.base_model.tp_k_head_num_, self.base_model.head_dim_),
                    dtype=torch.float16, device="cuda")
            b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index

        infer_state.init_some_extra_state(self.base_model, batch_size, total_token_num, max_len_in_batch,
                                          input_ids, b_loc, b_start_loc, b_seq_len, False)
        # print(f"\n\t<<Decode>>")
        # print(f"\t\tbatch_size {batch_size}")
        decode_start_time = time.time()
        predict_logics = self._token_forward(input_ids, infer_state, no_lora_compute, no_lora_copy)
        if self.use_sync:
            torch.cuda.synchronize()
        # print(f"\t<Decode end> --- time : {(decode_end_time - decode_start_time):0.8} ms -------------")
        self.timeDict["total_time"] = 1000 * (time.time() - decode_start_time)
        return predict_logics


    @final
    def _context_forward(self, input_ids, infer_state, no_lora_compute=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.context_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        self.timeDict["layers"] = []
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_context_forward(i, input_embs, infer_state, no_lora_compute)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _token_forward(self, input_ids, infer_state, no_lora_compute=False, no_lora_copy=False):
        cuda_input_ids = input_ids
        input_embs = self.base_model.pre_infer.token_forward(
                cuda_input_ids, infer_state, self.base_model.pre_post_weight)
        # print(f"\t\tInput embs : {input_embs.size()}")
        self.timeDict["layers"] = []
        for i in range(self.base_model.layers_num):
            input_embs = self._lora_token_forward(i, input_embs, infer_state, no_lora_compute, no_lora_copy)
        predict_logics = self.base_model.post_infer.token_forward(
                input_embs, infer_state, self.base_model.pre_post_weight, return_logics=True)
        return predict_logics


    @final
    def _lora_context_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        # print(f"\t\tLayer {layer_id}")
        
        attention_start_time = time.time()
        self._lora_context_attention(layer_id, input_embs, infer_state, no_lora_compute)
        
        if self.use_sync:
            torch.cuda.synchronize()
        attention_time = 1000 * (time.time() - attention_start_time)
        
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        
        ffn_start = time.time()
        layer_infer._context_ffn(input_embs, infer_state, layer_weight)
        
        if self.use_sync:
            torch.cuda.synchronize()
        
        ffn_time = 1000 * (time.time() - ffn_start)
        
        self.timeDict["layers"][-1]["total_attention_time"] = attention_time
        self.timeDict["layers"][-1]["total_ffn_time"] = ffn_time
        
        return input_embs


    @final
    # @calculate_time(show=True, min_cost_ms=0)
    def _lora_token_forward(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        # print(f"\t\tLayer {layer_id}")
        
        attention_start_time = time.time()
        self._lora_token_attention(layer_id, input_embs, infer_state, no_lora_compute, no_lora_copy)
        if self.use_sync:
            torch.cuda.synchronize()
        attention_time = 1000 * (time.time() - attention_start_time)
        
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        
        # mark_start("token_ffn")
        ffn_start = time.time()
        layer_infer._token_ffn(input_embs, infer_state, layer_weight)
        # mark_end("token_ffn")
        if self.use_sync:
            torch.cuda.synchronize()
        ffn_time = 1000 * (time.time() - ffn_start)
        
        self.timeDict["layers"][-1]["total_attention_time"] = attention_time
        self.timeDict["layers"][-1]["total_ffn_time"] = ffn_time
        
        return input_embs


    # @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _lora_context_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False):
        #print("\tAttention === ")
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]

        # layer normalization
        attention_norm_start = time.time()
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        attention_norm_time = 1000 * (time.time() - attention_norm_start)
        
        # fetch k, v 현재로는 그냥 infer_state.prefill_key_buffer, infer_state.prefill_value_buffer을 반환하는걸로 보임 (decode시 다름)
        precache_start = time.time()
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        precache_time = 1000 * (time.time() - precache_start)

        # gen new q, k, v (batch different adapters)
        get_qkv_start = time.time()
        q = self._lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute)
        if self.use_sync:
            pass
        torch.cuda.synchronize()
        get_qkv_time = 1000 * (time.time() - get_qkv_start)
        
        input1 = None

        postcache_start = time.time()
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        postcache_time = 1000 * (time.time() - postcache_start)
        
        # compute attention
        attention_start = time.time()
        o = layer_infer._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        attention_time = 1000 * (time.time() - attention_start)
        q = None
        
        get_o_start = time.time()
        o = self._lora_get_o(layer_id, o, infer_state, no_lora_compute)
        if self.use_sync:
            pass
        torch.cuda.synchronize()
        get_o_time = 1000 * (time.time() - get_o_start)
        
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        # residual
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        
        attention_time_dict = {}
        attention_time_dict["layer"]        = layer_id
        attention_time_dict["atten_norm"]   = attention_norm_time
        attention_time_dict["precache"]     = precache_time
        attention_time_dict["get_qkv"]      = {}
        attention_time_dict["get_qkv"]["total_time"] = get_qkv_time
        attention_time_dict["get_qkv"]["detail"] = self.get_qkv_timeDict
        attention_time_dict["postcache"]    = postcache_time
        attention_time_dict["atten_calc"]    = attention_time
        attention_time_dict["get_o"]        = {}
        attention_time_dict["get_o"]["total_time"] = get_o_time
        attention_time_dict["get_o"]["detail"] = self.get_o_timeDict
        
        self.timeDict["layers"].append(attention_time_dict)
        
        return


    # @calculate_time(show=True, min_cost_ms=0)
    # this impl dont to use @mark_cost_time
    def _lora_token_attention(self, layer_id, input_embs, infer_state, no_lora_compute=False, no_lora_copy=False):
        layer_weight = self.base_model.trans_layers_weight[layer_id]
        layer_infer = self.base_model.layers_infer[layer_id]
        
        # layer normalization
        attention_norm_start = time.time()
        input1 = layer_infer._att_norm(input_embs, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        attention_norm_time = 1000 * (time.time() - attention_norm_start)
        
        # fetch k, v
        precache_start = time.time()
        cache_k, cache_v = layer_infer._pre_cache_kv(infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        precache_time = 1000 * (time.time() - precache_start)
        
        # gen new q, k, v (batch different adapters)
        get_qkv_start = time.time()
        q = self._batch_lora_get_qkv(layer_id, input1, cache_k, cache_v, infer_state, no_lora_compute, no_lora_copy)
        if self.use_sync:
            torch.cuda.synchronize()
        get_qkv_time = 1000 * (time.time() - get_qkv_start)
        
        input1 = None
        
        postcache_start = time.time()
        layer_infer._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        postcache_time = 1000 * (time.time() - postcache_start)
        
        # compute attention
        
        attention_start = time.time()
        o = layer_infer._token_attention_kernel(q, infer_state, layer_weight)
        if self.use_sync:
            torch.cuda.synchronize()
        attention_time = 1000 * (time.time() - attention_start)
        
        q = None
        
        get_o_start = time.time()
        o = self._batch_lora_get_o(layer_id, o, infer_state, no_lora_compute)
        if self.use_sync:
            torch.cuda.synchronize()
        get_o_time = 1000 * (time.time() - get_o_start)
        
        # if self.world_size_ > 1:
        #     dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embs.add_(o.view(-1, layer_infer.embed_dim_))
        
        attention_time_dict = {}
        attention_time_dict["layer"]        = layer_id
        attention_time_dict["atten_norm"]   = attention_norm_time
        attention_time_dict["precache"]     = precache_time
        attention_time_dict["get_qkv"]      = {}
        attention_time_dict["get_qkv"]["total_time"] = get_qkv_time
        attention_time_dict["get_qkv"]["detail"] = self.get_qkv_timeDict
        attention_time_dict["postcache"]    = postcache_time
        attention_time_dict["atten_calc"]    = attention_time
        attention_time_dict["get_o"]        = {}
        attention_time_dict["get_o"]["total_time"] = get_o_time
        attention_time_dict["get_o"]["detail"] = self.get_o_timeDict
        
        self.timeDict["layers"].append(attention_time_dict)
        return
    

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False, no_lora_copy=False)->torch.Tensor:
        #print("\t\tBatch QKV")
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]

        #no_lora_compute = True

        # q (bs, H)
        base_start_Q = time.time()
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.q_weight_)
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_Q = 1000 * (time.time() - base_start_Q)

        # print(f"\t\tBase_layer embed_dim : {base_layer_infer.embed_dim_}")
        # print(f"\t\tQ's size : {q.shape}, is on cuda? {q.is_cuda}")
        # @TODO: fix me, filter requests querying only base model
        assert(len(q)==len(self.req_bins))

        if not no_lora_compute:
            # mark_start("get_q")
            lora_start_Q = time.time()
            delta_qA = self.delta[0]
            dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 0, self.infer_adapter.a_scaling)
            dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 0, self.infer_adapter.a_scaling)
            # delta_qA = None
            # mark_end("get_q")
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_Q = 1000 * (time.time() - lora_start_Q)
            
        rotary_emb_q_start = time.time()
        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                          infer_state.position_cos, infer_state.position_sin)
        if self.use_sync:
            torch.cuda.synchronize()
        rotary_emb_q_time = 1000 * (time.time() - rotary_emb_q_start)

        # k (bs, H)
        base_start_K = time.time()
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_K = 1000 * (time.time() - base_start_K)

        if not no_lora_compute:
            # mark_start("get_k")
            lora_start_K = time.time()
            delta_kA = self.delta[1]
            lora_start_Q = time.time()
            
            dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 1, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            # mark_end("get_k")
            
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_K = 1000 * (time.time() - lora_start_K)
            
        rotary_emb_k_start = time.time()
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        if self.use_sync:
            torch.cuda.synchronize()
        rotary_emb_k_time = 1000 * (time.time() - rotary_emb_k_start)

        # v (bs, H)
        base_start_V = time.time()
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_V = 1000 * (time.time() - base_start_V)
                
        if not no_lora_compute:
            # mark_start("get_v")
            lora_start_V = time.time()
            delta_vA = self.delta[2]
            dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 2, self.infer_adapter.a_scaling)
            dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                          delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
            # mark_end("get_v")
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_V = 1000 * (time.time() - lora_start_V)
            
        #printstart = time.time()
        # print(f"\t\tBase Q : {base_time_Q:.8f} ms | LoRA Q : {lora_time_Q:.8f} ms")
        # print(f"\t\tBase K : {base_time_K:.8f} ms | LoRA K : {lora_time_K:.8f} ms")
        # print(f"\t\tBase V : {base_time_V:.8f} ms | LoRA V : {lora_time_V:.8f} ms")
        self.get_qkv_timeDict = {}
        self.get_qkv_timeDict["q_base"] = base_time_Q
        if not no_lora_compute:
            self.get_qkv_timeDict["q_lora"] = lora_time_Q
        
            self.get_qkv_timeDict["q_lora_adapter_start"] = self.infer_adapter.a_start.numel()
            self.get_qkv_timeDict["q_lora_adapter_len"] = self.infer_adapter.a_len.numel()
            self.get_qkv_timeDict["q_lora_adapter_loc"] = self.infer_adapter.a_loc.numel()
            self.get_qkv_timeDict["q_lora_adapter_bins"] = self.req_bins.numel()
            self.get_qkv_timeDict["q_lora_adapter_scaling"] = self.infer_adapter.a_scaling.numel()
        
        self.get_qkv_timeDict["q_rotary_emb"] = rotary_emb_q_time
        self.get_qkv_timeDict["k_base"] = base_time_K
        if not no_lora_compute:
            self.get_qkv_timeDict["k_lora"] = lora_time_K
        self.get_qkv_timeDict["k_rotary_emb"] = rotary_emb_k_time
        self.get_qkv_timeDict["v_base"] = base_time_V
        if not no_lora_compute:
            self.get_qkv_timeDict["v_lora"] = lora_time_V
        #printend = time.time()
        #printtime = printend - printstart
        #print(f"\t\tprinttime : {printtime:.8f} ms | LoRA V : {printtime:.8f} ms")
        
        return q



    def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        
        #no_lora_compute = True
        
        base_start_Q = time.time()
        # with torch.cuda.stream(self.stream1):
        # q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
        #             base_layer_weight.q_weight_)

        if self.use_sync:
            torch.cuda.synchronize()
        base_time_Q = 1000 * (time.time() -  base_start_Q)
        
        #assert(len(q)==len(self.batch_req_bins))
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        if not no_lora_compute:
            # fix me: @TODO we need to filter out requests querying only base model
            delta_qA = self.delta[0]
            lora_start_Q = time.time()
            
            # if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # #if 1 == 0:
            #     lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
            #                                 self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
            #                                 delta_qA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
            #                                 self.infer_adapter.a_len, infer_state.b_start_loc, 
            #                                 infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
            #                                 0, self.max_lora_dim, self.max_b_seq_len)
            #     lora_get_qkvo_fwd_expand(delta_qA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
            #                                 q, self.infer_adapter.a_scaling, 
            #                                 self.infer_adapter.a_loc, self.infer_adapter.a_start, 
            #                                 self.infer_adapter.a_len, infer_state.b_start_loc, 
            #                                 infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
            #                                 0, self.max_lora_dim, self.max_b_seq_len)
            # else:
            # stream_pass_test()
            # stream_pass_test(self.stream1.cuda_stream)
            default_stream = torch.cuda.current_stream()
            # print(f"Default Stream: {default_stream.cuda_stream}")
            
            with torch.cuda.stream(self.stream1):
                torch.cuda.set_stream(self.stream1)
                # print(f"Base Stream: {self.stream1.cuda_stream}")
                # print(f"Changed Stream: {torch.cuda.current_stream().cuda_stream}")
                q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), # 1 * 409 #4096 * 4096
                         base_layer_weight.q_weight_)
                # print(input_embs.view(-1, base_layer_infer.embed_dim_).shape)
                # print(base_layer_weight.q_weight_.shape)
                # q = input_embs.view(-1, base_layer_infer.embed_dim_) @ base_layer_weight.q_weight_
            with torch.cuda.stream(self.stream2):
                torch.cuda.set_stream(self.stream2)
                # print(f"Shrink Stream: {self.stream2.cuda_stream}")
                # print(f"Changed Stream: {torch.cuda.current_stream().cuda_stream}")
                dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                        self.key_buffer[layer_id],
                        self.infer_adapter.a_start, self.infer_adapter.a_len, 
                        self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling, self.stream2.cuda_stream)
                
            self.stream1.synchronize()
            self.stream2.synchronize()
            dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                        self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                        self.batch_req_bins, 0, self.infer_adapter.a_scaling)
            
                # Batch gather matrix vector multiplication
            # delta_qA = None
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_Q = 1000 * (time.time() - lora_start_Q)
        else:
            with torch.cuda.stream(self.stream1):
                q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                        base_layer_weight.q_weight_)
            
        rotary_emb_q_start = time.time()
        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                       infer_state.position_cos, infer_state.position_sin)
        if self.use_sync:
            torch.cuda.synchronize()
        rotary_emb_q_time = 1000 * (time.time() - rotary_emb_q_start)


        # k (S, H)
        base_start_K = time.time()
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_K = 1000 * (time.time() - base_start_K)
        
        if not no_lora_compute:
            delta_kA = self.delta[1]
            lora_start_K = time.time()
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_kA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_kA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         1, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 1, self.infer_adapter.a_scaling)
            # delta_kA = None
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_K = 1000 * (time.time() - lora_start_K)
            
        rotary_emb_k_start = time.time()
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)
        if self.use_sync:
            torch.cuda.synchronize()
        rotary_emb_k_time = 1000 * (time.time() - rotary_emb_k_start)

        # v (S, H)
        base_start_V = time.time()
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_V = 1000 * (time.time() - base_start_V)
        
        if not no_lora_compute:
            delta_vA = self.delta[2]
            lora_start_V = time.time()
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64 and len(infer_state.b_seq_len) >= 2:
            # if 1 ==0:
                lora_get_qkvo_fwd_shrink(input_embs.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_vA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_vA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         cache_v.view(-1, base_model.tp_v_head_num_ * base_model.head_dim_), 
                                         self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, self.kv_embed_dim, 
                                         2, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
                dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                            delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
            
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_V = 1000 * (time.time() - lora_start_V)
            
        # printstart = time.time()
        # print(f"\t\tBase Q : {base_time_Q:.8f} ms | LoRA Q : {lora_time_Q:.8f} ms")
        # print(f"\t\tBase K : {base_time_K:.8f} ms | LoRA K : {lora_time_K:.8f} ms")
        # print(f"\t\tBase V : {base_time_V:.8f} ms | LoRA V : {lora_time_V:.8f} ms")
        
        self.get_qkv_timeDict = {}
        self.get_qkv_timeDict["q_base"] = base_time_Q
        if not no_lora_compute:
            self.get_qkv_timeDict["q_lora"] = lora_time_Q
        
            self.get_qkv_timeDict["q_lora_adapter_start"] = self.infer_adapter.a_start.numel()
            self.get_qkv_timeDict["q_lora_adapter_len"] = self.infer_adapter.a_len.numel()
            self.get_qkv_timeDict["q_lora_adapter_loc"] = self.infer_adapter.a_loc.numel()
            self.get_qkv_timeDict["q_lora_adapter_bins"] = self.batch_req_bins.numel()
            self.get_qkv_timeDict["q_lora_adapter_scaling"] = self.infer_adapter.a_scaling.numel()
        
        self.get_qkv_timeDict["q_rotary_emb"] = rotary_emb_q_time
        self.get_qkv_timeDict["k_base"] = base_time_K
        if not no_lora_compute:
            self.get_qkv_timeDict["k_lora"] = lora_time_K
        self.get_qkv_timeDict["k_rotary_emb"] = rotary_emb_k_time
        self.get_qkv_timeDict["v_base"] = base_time_V
        if not no_lora_compute:
            self.get_qkv_timeDict["v_lora"] = lora_time_V
        #printend = time.time()
        #printtime = printend - printstart
        #print(f"\t\tprinttime : {printtime:.8f} ms | LoRA V : {printtime:.8f} ms")
        return q
    

    # @calculate_time(show=True, min_cost_ms=0)
    def _batch_lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]    

        #no_lora_compute = True
        
        base_start_O = time.time()
        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_O = 1000 * (time.time() - base_start_O)
        
        if not no_lora_compute:
            # mark_start("get_o")
            delta_oA = self.delta[0]
            lora_start_O = time.time()
            dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                          self.key_buffer[layer_id], 
                          self.infer_adapter.a_start, self.infer_adapter.a_len, 
                          self.infer_adapter.a_loc, self.req_bins, 3, self.infer_adapter.a_scaling)
            dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                          self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                          self.req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
            # mark_end("get_o")
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_O = 1000 * (time.time() - lora_start_O)

        # print(f"\t\tBase O : {base_time_O:.8f} ms | LoRA O : {lora_time_O:.8f} ms")
        
        self.get_o_timeDict = {}
        self.get_o_timeDict["o_base"] = base_time_O
        if not no_lora_compute:
            self.get_o_timeDict["o_lora"] = lora_time_O
        return o


    def _lora_get_o(self, layer_id, input, infer_state, no_lora_compute=False)->torch.Tensor:
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]


        #no_lora_compute = True

        base_start_O = time.time()
        o = torch.mm(input.view(-1, base_layer_infer.embed_dim_),
                          base_layer_weight.o_weight_)
        
        if self.use_sync:
            torch.cuda.synchronize()
        base_time_O = 1000 * (time.time() - base_start_O)
        
        if not no_lora_compute:
            delta_oA = self.delta[0]
            lora_start_O = time.time()
            if self.max_b_seq_len >= 200 and self.max_lora_dim >= 64  and len(infer_state.b_seq_len) >= 2:
            # if 1 == 0:
                lora_get_qkvo_fwd_shrink(input.view(-1, base_layer_infer.embed_dim_), 
                                         self.key_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         delta_oA, self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
                lora_get_qkvo_fwd_expand(delta_oA, self.value_buffer[layer_id].view(-1, self.kv_embed_dim), 
                                         o, self.infer_adapter.a_scaling, 
                                         self.infer_adapter.a_loc, self.infer_adapter.a_start, 
                                         self.infer_adapter.a_len, infer_state.b_start_loc, 
                                         infer_state.b_seq_len, self.req_bins, base_layer_infer.embed_dim_, 
                                         3, self.max_lora_dim, self.max_b_seq_len)
            else:
                dispatch_bgmv(delta_oA, input.view(-1, base_layer_infer.embed_dim_), 
                            self.key_buffer[layer_id], 
                            self.infer_adapter.a_start, self.infer_adapter.a_len, 
                            self.infer_adapter.a_loc, self.batch_req_bins, 3, self.infer_adapter.a_scaling)
                dispatch_bgmv(o, delta_oA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                            self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                            self.batch_req_bins, 3, self.infer_adapter.a_scaling)
            # delta_oA = None
            if self.use_sync:
                torch.cuda.synchronize()
            lora_time_O = 1000 * (time.time() - lora_start_O)
            
        # print(f"\t\tBase O : {base_time_O:.8f} ms | LoRA O : {lora_time_O:.8f} ms")
        
        self.get_o_timeDict = {}
        self.get_o_timeDict["o_base"] = base_time_O
        if not no_lora_compute:
            self.get_o_timeDict["o_lora"] = lora_time_O
        return o


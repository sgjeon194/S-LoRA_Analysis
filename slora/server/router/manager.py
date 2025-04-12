import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue

from slora._kernels import dispatch_bgmv, stream_pass_test

import json
import shutil
import nvtx
from slora.server.router.stream_pool_manager import StreamPoolManager

def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0

        self.req_queue = get_scheduler(input_params, adapter_dirs)

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)

        self.time_dict_list = []
        self.warm_up_finished = False
        createStreamPoolManagerInstance = StreamPoolManager.instance()

    async def wait_to_model_ready(self):
        print("WAIT TO MODEL READY")
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))

        await asyncio.gather(*init_model_ret)
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        
        while True:
            await self._step()
            counter_count += 1
            #print(f"running batch {self.running_batch}")
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms


    async def loop_for_test_fwd(self):
        print("Start warm up")
        print(f"sync {self.input_params.use_sync}, no lora : {self.input_params.no_lora_compute}")
        
        self.req_queue.waiting_req_list = []
        
        batch_size = 4
        prompt_size = 1200 * batch_size
        token_num = 20
        all_lora_same = False
        for i in range(10):
            new_batch = self.req_queue.generate_test_batch(self.running_batch, self.lora_ranks, batch_size=batch_size, prompt_size=prompt_size, token_num=2, all_lora_same=all_lora_same)
            await self._step_prefill_test(new_batch)
            await self._decode_batch(self.running_batch)
            
        print("Decode end")
        print("warmup end\n")
        removeLogFile()
        
        print("Start test")
        start = time.time()        
        torch.cuda.nvtx.range_push("Real run")
        new_batch = self.req_queue.generate_test_batch(self.running_batch, self.lora_ranks, batch_size=batch_size, prompt_size=prompt_size, token_num=token_num, all_lora_same=all_lora_same)
        await self._step_prefill_test(new_batch)
        decode_start = time.time()
        for i in range(token_num - 1):
            await self._decode_batch(self.running_batch)
        torch.cuda.nvtx.range_pop()
        
        print(f"Decode end : {1000 * (time.time() - decode_start)}")
        print(f"Batch finished : {1000 * (time.time() - start)} ms")
        #moveLogFile("twostream.txt")
        #moveLogFile(f"batch_{batch_size}_token_{token_num}_useSync_{self.input_params.use_sync}_lora_{not self.input_params.no_lora_compute}.txt")
        #moveLogFile(f"batch_{batch_size}_token_{token_num}_useSync_{self.input_params.use_sync}_lora_{not self.input_params.no_lora_compute}_multistream.txt")
        removeLogFile()
        

    no_request_started = False
    
    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 이 함수는 서버가 켜져있는 동안 계속 호출되는 중
        if self.running_batch is None:
            req_queue_len = len(self.req_queue.waiting_req_list)
            # if req_queue_len < 2:
            #     return
            #new_batch = self.req_queue.generate_new_batch_equal_prompt_size(self.running_batch, self.lora_ranks)
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            # new_batch = self.req_queue.generate_new_batch_synthetic(self.running_batch, self.lora_ranks, 64, 8, 2)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                # print("\n==== Step1 - New batch ====")
                # print(f"\tInside request queue, there was {req_queue_len} requests")
                # print(f"\tThe new batch has {len(new_batch.reqs)} requests")
                # print(f"\tThe new batch uses {len(new_batch.adapter_dirs)} adapters")
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        # print(f"\tnew batch adapters : {str(new_batch.adapter_dirs)}")
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    print("peft")
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                print("Prefill end")
                self.has_wait_tokens = 0
                # exit()
                # print("---- Step1 end ----\n")
            else:
                if not self.no_request_started:
                    timeDict = {}
                    timeDict["batch_id"] = ""
                    timeDict["max_rank"] = 0
                    timeDict["ranks"] = 0
                    timeDict["request_num"] = 0
                    timeDict["adapter_num"] = 0
                    timeDict["total_token_num"] = 0
                    timeDict["max_len_in_batch"] = 0
                    timeDict["arithType"] = ""
                    timeDict["layers"] = []
                    timeDict["run_time"] = 1000 * time.time()
                    writeTimeDict(timeDict)
                    self.no_request_started = True
                    
            return
        
        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            
            #print("\n==== Step2 - Waiting for tokens ====")
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                print(f"\tCan generate next batch?? {next_batch is not None}")
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        print(f"22222adapters : {str(next_batch.adapter_dirs)}")
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            #print("---- Step2 end ----\n")
            return
        else:
            #print("\n==== Step3 - Even running a batch, can we generate more? ====")
            req_queue_len = len(self.req_queue.waiting_req_list)
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            #new_mini_batch = self.req_queue.generate_new_batch_equal_prompt_size(self.running_batch, self.lora_ranks)
            #new_mini_batch = self.req_queue.generate_new_batch_synthetic(self.running_batch, self.lora_ranks, 64, 8, 2)
            
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                print(f"\tInside request queue, there was {req_queue_len} requests")
                print(f"\tThere was {len(self.running_batch.reqs)} requests in the running batch")
                print(f"\tNow in the new minibatch there are {len(new_mini_batch.reqs)} requests")
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        print(f"\tMini batch adapters : {str(new_mini_batch.adapter_dirs)}")
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                    print(f"\tRunning batch now merged minibatch and there is now {len(self.running_batch.reqs)} requests in the running batch")
                self.has_wait_tokens = 0
            else:
                # print(f"\tRunning batch ")
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
            #print("---- Step3 end ----\n")
        
    async def _step_prefill_test(self, new_batch):
        
        if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
            self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
            self.req_queue.reset_abort_list()
        if new_batch is not None:
            self.running_batch = new_batch

            if not self.input_params.no_lora:
                # load adapters
                ret = []
                for tp_rank in range(self.world_size):
                    # print(f"\tnew batch adapters : {str(new_batch.adapter_dirs)}")
                    ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                await asyncio.gather(*ret)

            torch.cuda.synchronize()
            await self._prefill_batch(self.running_batch)
            await self._filter_runing_batch()
            print("Prefill end")
            self.has_wait_tokens = 0
        

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch, minibatch=True):
        await self._init_batch(batch)
        if self.no_request_started:
            timeDict = {}
            timeDict["batch_id"] = ""
            timeDict["max_rank"] = 0
            timeDict["ranks"] = 0
            timeDict["rank_types"] = []
            timeDict["request_num"] = 0
            timeDict["adapter_num"] = 0
            timeDict["total_token_num"] = 0
            timeDict["max_len_in_batch"] = 0
            timeDict["arithType"] = ""
            timeDict["layers"] = []
            timeDict["run_time"] = 1000 * time.time()
            writeTimeDict(timeDict)
            self.no_request_started = False
            
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0][0])
        else:
            req_to_out_token_id = ans[0][0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
            
        if self.warm_up_finished:
            timeDict = {}
            timeDict["batch_id"] = batch.batch_id
            # adapters = [req.adapter_dir for req in batch.reqs]
            # ranks = [self.lora_ranks[adapter] for adapter in adapters]
            
            ranks = [self.lora_ranks[adapter] for adapter in [req.adapter_dir for req in batch.reqs]]
            timeDict["max_rank"] = max(ranks)
            timeDict["ranks"] = sum(ranks)
            timeDict["rank_types"] = ranks
            
            timeDict.update(ans[0][1])
            writeTimeDict(timeDict)
            
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        return

    async def _decode_batch(self, batch:Batch):
        self.req_queue.update_counter(batch)
        
        if self.no_request_started:
            timeDict = {}
            timeDict["batch_id"] = ""
            timeDict["max_rank"] = 0
            timeDict["ranks"] = 0
            timeDict["rank_types"] = 0
            timeDict["request_num"] = 0
            timeDict["adapter_num"] = 0
            timeDict["total_token_num"] = 0
            timeDict["max_len_in_batch"] = 0
            timeDict["arithType"] = ""
            timeDict["layers"] = []
            timeDict["run_time"] = 1000 * time.time()
            writeTimeDict(timeDict)
            self.no_request_started = False
            
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        #print(f"Ans : len = {len(ans)} value sample = {ans[0]}")
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0][0])
        else:
            req_to_out_token_id = ans[0][0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        
            
        if self.warm_up_finished:
            timeDict = {}
            timeDict["batch_id"] = batch.batch_id
            ranks = [self.lora_ranks[adapter] for adapter in [req.adapter_dir for req in batch.reqs]]
            timeDict["max_rank"] = max(ranks)
            timeDict["ranks"] = sum(ranks)
            timeDict["rank_types"] = ranks
            timeDict.update(ans[0][1])
            writeTimeDict(timeDict)
            
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            batch.filter_finished()
            print("\n\tHas new finished request")
            print(f"Remaining requests : {len(self.req_queue.waiting_req_list)}")
            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                print("\tUnmerging Adapter (여기 체크 필요)")
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                print("\tOffloading adapter")
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)

            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                print("\tRunning batch is clear, so offload adpaters")
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)

            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
            if req.has_generate_finished:
                if self.warm_up_finished == False:
                    print("\nFinished warm up!\n")
                self.warm_up_finished = True
                
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            #print("\nRequest accepted!!\n")
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               use_sync = args.use_sync,
                               use_multistream = args.use_multistream
                              )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,            
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        # if input_params.profile:
        #     asyncio.run(router.profile_prefill())
        # if input_params.scheduler == "pets" and input_params.profile:
        #     router.req_queue.alpha = router.alpha_model
        #     router.req_queue.beta = router.beta_model
        # elif input_params.scheduler == "pets":
        #     # loading from file
        #     cache_dir = os.path.expanduser("~/.cache/slora")
        #     router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
        #     router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    # pipe_writer.send('init ok')
    
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.create_task(router.loop_for_fwd())
    # loop.run_until_complete(router.loop_for_netio_req())
    
    asyncio.run(router.loop_for_test_fwd())
    
    return

def writeTimeDict(data):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    folder_path = f"{project_root}/Logs"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file = open(f"{folder_path}/log.txt", 'a')
    file.write(json.dumps(data))
    file.write(", ")
    file.close()
    
def removeLogFile():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    file_path = f"{project_root}/Logs/log.txt"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        
def moveLogFile(destination_path):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    file_path = f"{project_root}/Logs/log.txt"
    destination_path = f"{project_root}/Logs/loraLatencyTest/rank32/{destination_path}"
    
    if os.path.exists(file_path):
        shutil.move(file_path, destination_path)
        
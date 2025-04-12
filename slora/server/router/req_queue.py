import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from slora.utils.infer_utils import calculate_time
import random

from ..sampling_params import SamplingParams

class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.static_id = 0
        
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False
    
    def update_counter(self, req):
        pass 

    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        #print(__name__)
        #print(f"Current batch size : {len(current_batch.reqs)}")
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None

    def generate_new_batch_equal_prompt_size(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if len(self.waiting_req_list) < 1 or current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        #print(__name__)
        #print(f"Current batch size : {len(current_batch.reqs)}")
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        
        batch_size = 1 #min(random.randint(1, len(self.waiting_req_list)), 32)
        total_batch_token_length = 1200
        
        new_input_length = self._equal_integer_divide(total_batch_token_length, batch_size)
        # token_per_req = total_batch_token_length // batch_size + 1
        # remain = batch_size - total_batch_token_length % batch_size
        # new_input_length = [a - b for a, b in zip([token_per_req for i in range(batch_size)], [1 if i < remain else 0 for i in range(batch_size)])]
        
        rank_64_loras = [name for name, rank in lora_ranks.items() if rank == 8]
        using_loras = random.choices(rank_64_loras, k=1)
        
        for idx in range(batch_size):
            req = self.waiting_req_list[idx]
            req.input_len = new_input_length[idx]
            if len(req.prompt_ids) < req.input_len:
                req.prompt_ids.extend([req.prompt_ids[-1]] * (req.input_len - len(req.prompt_ids)))
            else:
                req.prompt_ids = req.prompt_ids[:req.input_len]
        
        for idx in range(batch_size):
            req = self.waiting_req_list[idx]
            if req.aborted:
                aborted_count += 1
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                req.adapter_dir = using_loras[0]
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                return None

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None
    
    def generate_test_batch(self, current_batch:Batch, lora_ranks: dict[str, int], batch_size=1, prompt_size=1200, token_num=2, all_lora_same=False):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        
        request_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        
        lora_names = [name for name, _ in lora_ranks.items()]
        
        rank_64_loras = [name for name, rank in lora_ranks.items() if rank == 64]
        rank_32_loras = [name for name, rank in lora_ranks.items() if rank == 32]
        rank_16_loras = [name for name, rank in lora_ranks.items() if rank == 16]
        rank_8_loras = [name for name, rank in lora_ranks.items() if rank == 8]
        
        if not all_lora_same:
            using_ranks_64_loras = random.choices(rank_64_loras, k=0)
            using_ranks_32_loras = random.choices(rank_32_loras, k=0)
            using_ranks_16_loras = random.choices(rank_16_loras, k=0)
            using_ranks_8_loras = random.choices(rank_8_loras, k=batch_size)
            using_lora_adapters = using_ranks_64_loras + using_ranks_32_loras + using_ranks_16_loras + using_ranks_8_loras
        else:
            using_ranks_64_loras = random.choices(rank_64_loras, k=0)
            using_ranks_32_loras = random.choices(rank_32_loras, k=1)
            using_ranks_16_loras = random.choices(rank_16_loras, k=0)
            using_ranks_8_loras = random.choices(rank_8_loras, k=0)
            using_lora_adapters = using_ranks_64_loras + using_ranks_32_loras + using_ranks_16_loras + using_ranks_8_loras
            using_lora_adapters *= batch_size
        
        new_input_length = self._equal_integer_divide(prompt_size, batch_size)
        new_output_length = [token_num] * batch_size
               
        for idx in range(batch_size):
            prompt_ids = [1] + [15043] * (new_input_length[idx] - 1)
            new_sampling_params = SamplingParams(False, 0.0, 0.0, 1.0, 1.0, 1, True, new_output_length[idx], [])
            new_sampling_params.max_new_tokens = new_output_length[idx]
            request_list.append(Req(using_lora_adapters[idx], self.static_id, prompt_ids, new_sampling_params))
            self.static_id = self.static_id + 1
        
        can_run_list = []
        for req in request_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                return None

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[1:]
            return new_batch
        else:
            return None
        
    def generate_new_batch_synthetic(self, current_batch:Batch, lora_ranks: dict[str, int], rank_a, rank_b, rank_ratio):
        if len(self.waiting_req_list) < 1 or current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        request_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        
        rank_a_loras = [name for name, rank in lora_ranks.items() if rank == rank_a]
        rank_b_loras = [name for name, rank in lora_ranks.items() if rank == rank_b]
        
        rank_a_lora_num = 4#random.randint(rank_ratio, 32) // rank_ratio
        ranks_a = random.choices(rank_a_loras, k=rank_a_lora_num)
        rank_b_lora_num = rank_a_lora_num * rank_ratio
        ranks_b = random.choices(rank_b_loras, k=rank_b_lora_num)
        
        using_lora_adapters = ranks_a + ranks_b
        batch_size = len(using_lora_adapters)
        random.shuffle(using_lora_adapters)
        new_input_length = self._random_integer_divide(1200, batch_size)
        new_output_length = [random.randint(8, 512) for i in range(batch_size)]
        
        first_token = self.waiting_req_list[0].prompt_ids[0]
        dummy_token = self.waiting_req_list[0].prompt_ids[-1]
        
        for idx in range(batch_size):
            prompt_ids = [first_token] + [dummy_token] * (new_input_length[idx] - 1)
            new_sampling_params = self.waiting_req_list[0].sample_params
            new_sampling_params.max_new_tokens = new_output_length[idx]
            request_list.append(Req(using_lora_adapters[idx], self.static_id, prompt_ids, self.waiting_req_list[0].sample_params))
            self.static_id = self.static_id + 1
                
        can_run_list = []
        for req in request_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                return None

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[1:]
            return new_batch
        else:
            return None


    def _equal_integer_divide(self, total, n):
        token_per_req = total // n + 1
        remain = n - total % n
        new_input_length = [a - b for a, b in zip([token_per_req for i in range(n)], [1 if i < remain else 0 for i in range(n)])]
        return new_input_length
        

    def _random_integer_divide(self, total, n):
        min_value = 8
        remaining_total = total - n * min_value
        
        if remaining_total < 0:
            raise ValueError("총합이 너무 작아서 값을 분배할 수 없습니다.")
        
        if n == 0:
            print("pass")
            pass
        random_ratios = np.random.rand(n)
        random_ratios /= np.sum(random_ratios)
        values = np.floor(random_ratios * remaining_total).astype(int)
        diff = remaining_total - np.sum(values)
        
        for i in range(diff):
            values[i % n] += 1
        
        values += min_value
        
        return values.tolist()


    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            return next_batch
        else:
            return None

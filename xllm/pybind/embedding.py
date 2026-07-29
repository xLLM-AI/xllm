# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import signal
import sys
import time
from . import utils
from typing import Any, List, Optional, Union

from xllm_export import (LLMMaster, Options, RequestOutput,
                         RequestParams)
from .errors import ValidationError

class Embedding:
    def __init__(
        self,
        model: str,
        limit_image_per_prompt: int = 8,
        block_size: int = 128,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.8,
        enable_prefix_cache: bool = True,
        max_tokens_per_batch: int = 10240,
        max_seqs_per_batch: int = 1024,
        max_tokens_per_chunk_for_prefill: int = -1,
        speculative_algorithm: str = 'MTP',
        num_request_handling_threads: int = 4,
        communication_backend: str = 'hccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        enable_chunked_prefill: bool = True,
        instance_role: str = 'DEFAULT',
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        enable_schedule_overlap: bool = False,
        enable_graph: bool = False,
        enable_graph_mode_decode_no_padding: bool = False,
        enable_prefill_piecewise_graph: bool = False,
        max_tokens_for_graph_mode: int = 2048,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        use_cpp_chat_template: bool = True,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        model_type = utils._infer_model_type(model)

        options = Options()
        options.model_path = model
        options.task_type = "embed"
        options.draft_model_path = None
        options.backend = "llm"
        options.limit_image_per_prompt = limit_image_per_prompt
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.max_tokens_per_chunk_for_prefill = max_tokens_per_chunk_for_prefill
        options.speculative_algorithm = speculative_algorithm
        options.num_request_handling_threads = num_request_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_chunked_prefill = enable_chunked_prefill
        free_port = utils.get_free_port()
        options.master_node_addr = "127.0.0.1:" + str(free_port)
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.cp_size = cp_size
        options.ep_size = ep_size
        options.enable_disagg_pd = False
        options.enable_schedule_overlap = enable_schedule_overlap
        options.enable_graph = enable_graph
        options.enable_graph_mode_decode_no_padding = enable_graph_mode_decode_no_padding
        options.enable_prefill_piecewise_graph = enable_prefill_piecewise_graph
        options.max_tokens_for_graph_mode = max_tokens_for_graph_mode
        options.enable_offline_inference = True
        options.spawn_worker_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        options.enable_shm = enable_shm
        options.is_local = is_local
        options.input_shm_size = input_shm_size
        options.output_shm_size = output_shm_size
        utils._configure_cpp_chat_template(use_cpp_chat_template, model_type)
        self.master = LLMMaster(options)

    def finish(self) -> None:
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            utils.terminate_process(os.getpid())
        except Exception as e:
            pass

    def embedding(
        self,
        inputs: Union[str, List[str]],
        request_params: Optional[Union[RequestParams, List[RequestParams]]] = None,
        wait_for_schedule: bool = True,
    ) -> List[RequestOutput]:
        if request_params is None:
            request_params = RequestParams()
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(request_params, RequestParams):
            request_params.is_embeddings = True
            request_params = [request_params]
        else:
            for i in range(len(request_params)):
                request_params[i].is_embeddings = True

        outputs = [None] * len(inputs)
        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # schedule all requests
        self.master.handle_batch_request(
            inputs, request_params, callback
        )

        # TODO: add wait later
        if wait_for_schedule:
            pass

        # generate
        self.master.generate()

        # wait async output
        for i in range(len(outputs)):
            while outputs[i] is None:
                time.sleep(0.01)
            if outputs[i].status is not None and not outputs[i].status.ok:
                raise ValidationError(outputs[i].status.code, outputs[i].status.message)
            outputs[i].prompt = inputs[i]

        return outputs

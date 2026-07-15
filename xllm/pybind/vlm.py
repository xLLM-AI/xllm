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
import threading
from . import utils
from typing import Any, Callable, Dict, List, Optional, Union
from xllm_export import (VLMMaster, Options, RequestOutput,
                         RequestParams, MMData)
from .errors import ValidationError
from .params import (
    SamplingParams,
    to_request_params_list,
)


def _get_tqdm(
    use_tqdm: Union[bool, Callable[..., Any]],
) -> Optional[Callable[..., Any]]:
    if not use_tqdm:
        return None
    if callable(use_tqdm):
        return use_tqdm
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required when use_tqdm=True. "
            "Set use_tqdm=False to disable the progress bar."
        ) from exc
    return tqdm


class VLM:
    def __init__(
        self,
        model: str,
        task: str = "generate",
        draft_model: Optional[str] = '',
        limit_image_per_prompt: int = 8,
        block_size: int = 128,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.8,
        enable_prefix_cache: bool = True,
        max_encoder_cache_size: int = 0,
        max_tokens_per_batch: int = 10240,
        max_seqs_per_batch: int = 1024,
        max_tokens_per_chunk_for_prefill: int = -1,
        num_speculative_tokens: int = 0,
        speculative_algorithm: str = 'MTP',
        num_request_handling_threads: int = 4,
        communication_backend: str = 'hccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        enable_chunked_prefill: bool = True,
        instance_role: str = 'DEFAULT',
        transfer_listen_port: int = 26000,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        instance_name: str = '',
        enable_disagg_pd: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = 'PUSH',
        enable_graph: bool = False,
        enable_graph_mode_decode_no_padding: bool = False,
        enable_prefill_piecewise_graph: bool = False,
        max_tokens_for_graph_mode: int = 2048,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        use_cpp_chat_template: bool = True,
        disable_log_stats: bool = True,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")
        self.model = model
        model_type = utils._infer_model_type(model)

        options = Options()
        options.model_path = model
        options.task_type = task
        options.draft_model_path = draft_model
        options.backend ="vlm"
        options.limit_image_per_prompt = limit_image_per_prompt
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
        options.max_encoder_cache_size = max_encoder_cache_size
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.max_tokens_per_chunk_for_prefill = max_tokens_per_chunk_for_prefill
        options.num_speculative_tokens = num_speculative_tokens
        options.speculative_algorithm = speculative_algorithm
        options.num_request_handling_threads = num_request_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_chunked_prefill = enable_chunked_prefill
        free_port = utils.get_free_port()
        options.master_node_addr = "127.0.0.1:" + str(free_port)
        options.transfer_listen_port = transfer_listen_port
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.cp_size = cp_size
        options.ep_size = ep_size
        options.instance_name = instance_name
        options.enable_disagg_pd = enable_disagg_pd
        options.enable_schedule_overlap = enable_schedule_overlap
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
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
        options.disable_log_stats = disable_log_stats
        utils._configure_cpp_chat_template(use_cpp_chat_template, model_type)
        self.master = VLMMaster(options)

    def finish(self) -> None:
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            utils.terminate_process(os.getpid())
        except Exception as e:
            pass

    def generate(
        self,
        prompts: Union[
            str,
            List[str],
            Dict[str, Any],
            List[Dict[str, Any]],
        ],
        sampling_params: Optional[Union[
            SamplingParams,
            List[SamplingParams],
        ]] = None,
        wait_for_schedule: bool = True,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        from . import mm_utils
        prompts, mm_datas, image_urls = mm_utils.normalize_vllm_style_inputs(prompts)

        request_params = kwargs.pop("request_params", None)
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if request_params is None:
            request_params = sampling_params
        elif sampling_params is not None:
            raise ValueError("sampling_params and request_params cannot both be set")

        request_params_list = to_request_params_list(
            request_params, default_cls=SamplingParams
        )
        if len(request_params_list) not in (1, len(prompts)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the number of prompts."
            )

        outputs = [None] * len(prompts)
        progress_bar = None
        progress_bar_lock = threading.Lock()
        tqdm_cls = _get_tqdm(use_tqdm)
        if tqdm_cls is not None:
            progress_bar = tqdm_cls(total=len(prompts), desc="Processed prompts")

        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            if progress_bar is not None:
                with progress_bar_lock:
                    progress_bar.update(1)
            return True

        try:
            # schedule the batch requests
            if image_urls is not None:
                self.master.handle_batch_request_with_image_urls(
                    prompts, image_urls, request_params_list, callback
                )
            else:
                self.master.handle_batch_request(
                    prompts, mm_datas, request_params_list, callback
                )

            # wait for batch request to be scheduled
            if wait_for_schedule:
                pass

            # run until all scheduled requsts complete
            self.master.generate()

            # throw an exception if there is any error
            for index, output in enumerate(outputs):
                if output is None:
                    raise RuntimeError("Request failed, no output received")
                if output.status is not None and not output.status.ok:
                    raise ValidationError(output.status.code, output.status.message)
                # carry over the prompt to the output
                output.prompt = prompts[index]
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return outputs

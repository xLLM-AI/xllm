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
import time
import uuid
from . import utils
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from xllm_export import (LLMMaster, VLMMaster, Options, RequestOutput,
                         RequestParams)
from .errors import ValidationError
from .params import (
    BeamSearchParams,
    PoolingParams,
    SamplingParams,
    _RequestParamsProxy,
    to_request_params,
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


class BeamSearchOutput:
    def __init__(self, output: RequestOutput) -> None:
        self.prompt = output.prompt
        self.sequences = output.outputs
        self.status = output.status
        self.usage = output.usage
        self.request_output = output


class EmbeddingOutputs:
    def __init__(self, output: RequestOutput) -> None:
        embedding = []
        if output.outputs and len(output.outputs) > 0:
            embedding = output.outputs[0].embeddings
        self.embedding = embedding
        self.embeddings = embedding


class EmbeddingOutput:
    def __init__(self, output: RequestOutput) -> None:
        self.prompt = output.prompt
        self.outputs = EmbeddingOutputs(output)
        self.status = output.status
        self.usage = output.usage
        self.request_output = output


class LLM:
    @staticmethod
    def _is_vllm_style_inputs(prompts: object) -> bool:
        if isinstance(prompts, dict):
            return True
        if isinstance(prompts, list) and prompts and all(isinstance(x, dict) for x in prompts):
            return True
        return False

    def __init__(
        self,
        model: str,
        task: str = "generate",
        runner: Optional[str] = None,
        draft_model: Optional[str] = '',
        limit_image_per_prompt: int = 8,
        block_size: int = 128,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.8,
        enable_prefix_cache: bool = True,
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
        master_node_addr: str = '',
        instance_role: str = 'DEFAULT',
        transfer_listen_port: int = 26000,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        instance_name: str = '',
        enable_disagg_pd: bool = False,
        enable_pd_ooc: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = 'PUSH',
        disable_ttft_profiling: bool = False,
        enable_forward_interruption: bool = False,
        enable_graph: bool = False,
        enable_graph_mode_decode_no_padding: bool = False,
        enable_prefill_piecewise_graph: bool = False,
        max_tokens_for_graph_mode: int = 2048,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        kv_cache_dtype: str = 'auto',
        use_cpp_chat_template: bool = True,
        disable_log_stats: bool = True,
        enable_sleep_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        if runner is not None:
            if runner != "pooling":
                raise ValueError(f"unsupported runner: {runner}")
            task = "embed"

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        model_type, backend = utils._infer_model_type_and_backend(model)
        if backend == "dit":
            raise ValueError("LLM does not support DiT backend models")
        if model_type is None:
            raise ValueError("model_type is required for offline inference")
        utils._configure_cpp_chat_template(use_cpp_chat_template, model_type)

        options = Options()
        options.model_path = model
        options.task_type = task
        options.draft_model_path = draft_model
        options.backend = backend
        options.limit_image_per_prompt = limit_image_per_prompt
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = enable_prefix_cache
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
        if master_node_addr:
            options.master_node_addr = master_node_addr
        else:
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
        options.enable_pd_ooc = enable_pd_ooc
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
        options.disable_ttft_profiling = disable_ttft_profiling
        options.enable_forward_interruption = enable_forward_interruption
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
        options.kv_cache_dtype = kv_cache_dtype
        options.disable_log_stats = disable_log_stats
        options.enable_sleep_mode = enable_sleep_mode
        self._backend = backend
        if backend == "vlm":
            self.master = VLMMaster(options)
        else:
            self.master = LLMMaster(options)

    def finish(self) -> None:
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            utils.terminate_process(os.getpid())
        except Exception as e:
            pass

    def sleep(self) -> None:
        """Deep sleep: release device HBM (model weights + KV cache) for RL
        training, returning the physical memory to the driver.

        Contents are discarded; after ``wake_up`` weights are re-loaded and
        KV cache is re-prefilled. Requires the engine to be created with
        ``enable_sleep_mode=True``.
        """
        self.master.sleep()

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """Re-acquire device HBM previously released by ``sleep``.

        ``tags`` is reserved for future fine-grained wake-up
        (e.g. ["weights"] / ["kv_cache"]); the current version performs a
        full wake-up regardless of ``tags``.
        """
        self.master.wake_up()

    def is_sleeping(self) -> bool:
        return self.master.is_sleeping()

    def update_weights(self, checkpoint_path: str = "") -> None:
        """Reload model weights in place from disk (vllm-ascend ``reload_weights``).

        Typical level-2 RL flow:
            llm.sleep()
            llm.wake_up()                 # re-map empty weight + KV memory
            llm.update_weights("/path/to/new_ckpt")  # reload weights in place
        An empty ``checkpoint_path`` reuses the original model path.
        Requires ``enable_sleep_mode=True``.
        """
        self.master.update_weights(checkpoint_path)

    def generate(
        self,
        prompts: Union[
            str,
            List[str],
            Dict[str, object],
            List[Dict[str, object]],
        ],
        sampling_params: Optional[Union[
            SamplingParams,
            List[SamplingParams],
        ]] = None,
        wait_for_schedule: bool = True,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        request_params = kwargs.pop("request_params", None)
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if request_params is None:
            request_params = sampling_params
        elif sampling_params is not None:
            raise ValueError("sampling_params and request_params cannot both be set")

        mm_datas = None
        image_urls = None
        if self._is_vllm_style_inputs(prompts):
            from . import mm_utils
            prompts, mm_datas, image_urls = mm_utils.normalize_vllm_style_inputs(prompts)
        else:
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
                raise TypeError("prompts must be str/list[str] or vLLM-style dicts")

        request_params_list = to_request_params_list(
            request_params, default_cls=SamplingParams)
        if len(request_params_list) not in (1, len(prompts)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the "
                "number of prompts."
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
            # schedule all requests
            if self._backend == "vlm":
                if mm_datas is not None:
                    self.master.handle_batch_request(
                        prompts, mm_datas, request_params_list, callback
                    )
                else:
                    if image_urls is None:
                        image_urls = [[] for _ in prompts]
                    self.master.handle_batch_request_with_image_urls(
                        prompts, image_urls, request_params_list, callback
                    )
            else:
                has_images = image_urls is not None and any(image_urls)
                if mm_datas is not None or has_images:
                    raise ValueError("multi_modal_data is only supported for VLM models")
                self.master.handle_batch_request(
                    prompts, request_params_list, callback
                )

            # TODO: add wait later
            if wait_for_schedule:
                pass

            # generate
            self.master.generate()

            count = len(prompts)
            idx = 0
            while idx < count:
                # wait async output
                if outputs[idx] is None:
                    continue
                if outputs[idx].status is not None and not outputs[idx].status.ok:
                    raise ValidationError(outputs[idx].status.code, outputs[idx].status.message)
                outputs[idx].prompt = prompts[idx]
                idx += 1
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return outputs

    def beam_search(
        self,
        prompts: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
        params: Optional[Union[RequestParams, BeamSearchParams]] = None,
        wait_for_schedule: bool = True,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
    ) -> List[BeamSearchOutput]:
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]

        parsed_prompts: List[str] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                parsed_prompts.append(prompt)
                continue
            if isinstance(prompt, dict):
                if "prompt" not in prompt:
                    raise ValueError("beam_search prompt dict must contain key 'prompt'")
                parsed_prompts.append(prompt["prompt"])
                continue
            raise TypeError("prompts must be str or dict with key 'prompt'")

        explicit_fields = (
            params.explicit_fields()
            if isinstance(params, _RequestParamsProxy)
            else set()
        )
        params = to_request_params(params, default_cls=BeamSearchParams)
        if params.beam_width <= 0:
            raise ValueError("beam_width must be greater than 0")
        elif params.beam_width > 1:
            # Beam search relies on top-k logprob candidates from sampler.
            # Keep this aligned with the LLM request-path default.
            if "logprobs" not in explicit_fields:
                params.logprobs = True
            if (
                "top_logprobs" not in explicit_fields
                and params.top_logprobs == 0
            ):
                params.top_logprobs = params.beam_width

        outputs = self.generate(parsed_prompts,
                                request_params=params,
                                wait_for_schedule=wait_for_schedule,
                                use_tqdm=use_tqdm)
        return [BeamSearchOutput(output) for output in outputs]

    def embed(
        self,
        prompts: Union[str, List[str]],
        pooling_params: Optional[Union[
            RequestParams,
            PoolingParams,
            List[Union[RequestParams, PoolingParams]],
        ]] = None,
        wait_for_schedule: bool = True,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
    ) -> List[EmbeddingOutput]:
        request_params_list = to_request_params_list(
            pooling_params, default_cls=PoolingParams)
        for params in request_params_list:
            params.is_embeddings = True

        use_params: Union[RequestParams, List[RequestParams]]
        if len(request_params_list) == 1:
            use_params = request_params_list[0]
        else:
            use_params = request_params_list

        outputs = self.generate(prompts,
                                request_params=use_params,
                                wait_for_schedule=wait_for_schedule,
                                use_tqdm=use_tqdm)
        return [EmbeddingOutput(output) for output in outputs]

    @staticmethod
    def _normalize_selector_values(
        prompts: Sequence[str],
        selector: Union[str, dict, Sequence[Union[str, dict]]],
    ) -> List[str]:
        def get_literal(value: Union[str, dict]) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                selector_type = value.get("type", "literal")
                literal = value.get("value", "")
                if selector_type != "literal":
                    raise ValueError("selector.type must be literal")
                if not isinstance(literal, str) or not literal:
                    raise ValueError("selector.value is required")
                return literal
            raise ValueError("selector must be a string or dict")

        if isinstance(selector, (str, dict)):
            literal = get_literal(selector)
            return [literal for _ in prompts]

        selector_values = list(selector)
        if len(selector_values) != len(prompts):
            raise ValueError("selector count must match prompts count")
        return [get_literal(item) for item in selector_values]

    @staticmethod
    def _build_request_params_list(
        prompts: Sequence[str],
        request_params: Optional[Union[RequestParams, Sequence[RequestParams]]],
    ) -> List[RequestParams]:
        if request_params is None:
            return [RequestParams() for _ in prompts]
        if isinstance(request_params, RequestParams):
            if len(prompts) != 1:
                raise ValueError(
                    "request_params must be a list when prompts has multiple items"
                )
            return [request_params]

        params_list = list(request_params)
        if len(params_list) != len(prompts):
            raise ValueError("request_params count must match prompts count")
        return params_list

    def sample(
        self,
        prompts: Union[str, List[str]],
        selector: Union[str, dict, Sequence[Union[str, dict]]],
        request_params: Optional[Union[RequestParams, Sequence[RequestParams]]] = None,
        logprobs: int = 5,
        wait_schedule_done: bool = True,
    ) -> List[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            return []

        selector_values = self._normalize_selector_values(prompts, selector)
        params_list = self._build_request_params_list(prompts, request_params)
        if len(params_list) > 1:
            # sample() 会原地修改每个 RequestParams（如 request_id/sample_slots）。
            # 若复用同一个对象，会在并发批处理时互相覆盖。
            unique_param_objects = {id(p) for p in params_list}
            if len(unique_param_objects) != len(params_list):
                raise ValueError(
                    "request_params contains duplicated RequestParams objects. "
                    "Please create one RequestParams instance per prompt."
                )

        for i, prompt in enumerate(prompts):
            params = params_list[i]
            if not params.request_id:
                params.request_id = "sample-" + uuid.uuid4().hex

            params.max_tokens = 1
            params.n = 1
            params.best_of = 1
            params.logprobs = True
            params.top_logprobs = logprobs
            params.add_special_tokens = True
            params.is_sample_request = True

            ok, sample_slots = self.master.build_sample_slots(
                params.request_id,
                prompt,
                selector_values[i],
            )
            if not ok:
                raise ValueError(
                    "Failed to build sample slots. "
                    "selector.value must be a stable single special token."
                )
            params.sample_slots = sample_slots

        outputs = [None] * len(prompts)

        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        self.master.handle_batch_request(prompts, params_list, callback)

        if wait_schedule_done:
            pass

        self.master.generate()

        for i in range(len(outputs)):
            while outputs[i] is None:
                time.sleep(0.01)
            if outputs[i].status is not None and not outputs[i].status.ok:
                raise RuntimeError(
                    f"sample request failed: {outputs[i].status.message}"
                )
            outputs[i].prompt = prompts[i]

        return outputs

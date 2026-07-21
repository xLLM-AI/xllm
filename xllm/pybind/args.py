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

import argparse
from argparse import Namespace


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "t", "1", "yes", "y", "on"):
        return True
    if value in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class ArgumentParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=str, help='"Name or path of the huggingface model to use."')
        self.parser.add_argument('--task', type=str, default="generate", help='The task to use the model for. generate/embed.')
        self.parser.add_argument('--runner', type=str, choices=['pooling'], default=None, help='Optional runner mode for LLM. Currently supports: pooling.')
        self.parser.add_argument('--draft_model', type=str, default='', help='draft hf model path to the model file.')
        self.parser.add_argument('--limit_image_per_prompt', type=int, default=8, help='Maximum number of images per prompt.')
        self.parser.add_argument('--block_size', type=int, default=128, help='Number of slots per kv cache block. Default is 128.')
        self.parser.add_argument('--max_cache_size', type=int, default=0, help='Max gpu memory size for kv cache. Default is 0, which means cache size is calculated by available memory.')
        self.parser.add_argument('--max_memory_utilization', type=float, default=0.8, help='The fraction of GPU memory to be used for model inference, including model weights and kv cache.')
        self.parser.add_argument('--enable_prefix_cache', nargs='?', const=True, default=True, type=_str_to_bool, help='Whether to enable the prefix cache for the block manager.')
        self.parser.add_argument('--max_tokens_per_batch', type=int, default=10240, help='Max number of tokens per batch.')
        self.parser.add_argument('--max_seqs_per_batch', type=int, default=1024, help='Max number of sequences per batch.')
        self.parser.add_argument('--max_tokens_per_chunk_for_prefill', type=int, default=-1, help='Max number of tokens per chunk for request in prefill stage.')
        self.parser.add_argument('--num_speculative_tokens', type=int, default=0, help='Number of speculative tokens.')
        self.parser.add_argument('--speculative_algorithm', type=str, default='MTP', help='Speculative decoding algorithm. Supported options: MTP, Eagle3, Suffix, DFlash.')
        self.parser.add_argument('--num_request_handling_threads', type=int, default=4, help='Number of handling threads.')
        self.parser.add_argument('--communication_backend', type=str, default='hccl', help='npu communication backend.')
        self.parser.add_argument('--rank_tablefile', type=str, default='', help='atb hccl rank table file')
        self.parser.add_argument('--expert_parallel_degree', type=int, default=0, help='ep degree')
        self.parser.add_argument('--enable_chunked_prefill', nargs='?', const=True, default=True, type=_str_to_bool, help='Whether to enable chunked prefill.')
        self.parser.add_argument('--master_node_addr', type=str, default='', help='The master address for multi-node distributed serving(e.g. 10.18.1.1:9999).')
        self.parser.add_argument('--instance_role', type=str, default='DEFAULT', help='The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX).')
        self.parser.add_argument('--transfer_listen_port', type=int, default=26000, help='The KVCacheTranfer listen port.')
        self.parser.add_argument('--nnodes', type=int, default=1, help='The number of multi-nodes.')
        self.parser.add_argument('--node_rank', type=int, default=0, help='The node rank.')
        self.parser.add_argument('--dp_size', type=int, default=1, help='Data parallel size for MLA attention.')
        self.parser.add_argument('--cp_size', type=int, default=1, help='Context parallel size. Backend and model support is required.')
        self.parser.add_argument('--ep_size', type=int, default=1, help='Expert parallel size for MoE model.')
        self.parser.add_argument('--instance_name', type=str, default='', help='instance name')
        self.parser.add_argument('--enable_disagg_pd', nargs='?', const=True, default=False, type=_str_to_bool, help='Enable disaggregated prefill and decode execution.')
        self.parser.add_argument('--enable_pd_ooc', nargs='?', const=True, default=False, type=_str_to_bool, help='Enable online-offline co-location in disaggregated prefill-decoding mode.')
        self.parser.add_argument('--enable_schedule_overlap', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to enable schedule overlap.')
        self.parser.add_argument('--kv_cache_transfer_mode', type=str, default='PUSH', help='The mode of kv cache transfer(e.g. PUSH, PULL).')
        self.parser.add_argument('--disable_ttft_profiling', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to disable TTFT profiling.')
        self.parser.add_argument('--enable_forward_interruption', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to enable forward interruption.')
        self.parser.add_argument('--enable_graph', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to enable graph execution for offline inference.')
        self.parser.add_argument('--enable_graph_mode_decode_no_padding', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to enable graph-mode decode without padding for offline inference.')
        self.parser.add_argument('--enable_prefill_piecewise_graph', nargs='?', const=True, default=False, type=_str_to_bool, help='Whether to enable prefill piecewise graph for offline inference.')
        self.parser.add_argument('--max_tokens_for_graph_mode', type=int, default=2048, help='Maximum number of tokens for graph execution.')
        self.parser.add_argument('--enable_shm', nargs='?', const=True, default=False, type=_str_to_bool, help='Use shared memory for inter-process communication in the single-machine multi-GPU scenario.')
        self.parser.add_argument('--input_shm_size', type=int, default=1024, help='The size of input shared memory in MB.')
        self.parser.add_argument('--output_shm_size', type=int, default=128, help='The size of output shared memory in MB.')
        self.parser.add_argument('--kv_cache_dtype', type=str, default='auto', help='KV cache data type. "auto" (default) aligns with model dtype, "int8" enables INT8 quantization (MLU only).')
        self.parser.add_argument('--use_cpp_chat_template', nargs='?', const=True, default=True, type=_str_to_bool, help='Whether to use native C++ chat template for supported models.')
        self.parser.add_argument('--disable_log_stats', nargs='?', const=True, default=True, type=_str_to_bool, help='Whether to disable per-request statistic logs in offline inference.')

    def parse_args(self) -> Namespace:
        return self.parser.parse_args()

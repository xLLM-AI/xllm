/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLLM_LLM_DEFAULT_H
#define XLLM_LLM_DEFAULT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

const XLLM_InitOptions XLLM_INIT_LLM_OPTIONS_DEFAULT = {
    .enable_chunked_prefill = false,
    .enable_prefill_sp = false,
    .enable_prefix_cache = false,
    .enable_disagg_pd = false,
    .enable_pd_ooc = false,
    .enable_schedule_overlap = false,
    .enable_shm = false,

    .transfer_listen_port = 26000,
    .nnodes = 1,
    .node_rank = 0,
    .dp_size = 1,
    .ep_size = 1,
    .block_size = 32,
    .max_cache_size = 0,
    .max_tokens_per_batch = 20480,
    .max_seqs_per_batch = 256,
    .max_tokens_per_chunk_for_prefill = 0,
    .num_speculative_tokens = 0,
    .num_request_handling_threads = 4,
    .expert_parallel_degree = 0,
    .server_idx = 0,
    .max_memory_utilization = 0.9,
    .flashinfer_workspace_buffer_size = 134217728,
    .request_queue_size = 0,
    .task = "generate",
    .communication_backend = "lccl",
    .instance_role = "DEFAULT",
    .device_ip = "",
    .master_node_addr = "127.0.0.1:18899",
    .xservice_addr = "",
    .instance_name = "",
    .kv_cache_transfer_mode = "PUSH",
    .log_dir = "",
    .draft_model = "",
    .draft_devices = "",
    .cpu_affinity = ""};

const XLLM_RequestParams XLLM_LLM_REQUEST_PARAMS_DEFAULT = {
    .echo = false,
    .offline = false,
    .logprobs = false,
    .ignore_eos = false,

    .n = 1,
    .max_tokens = 5120,
    .best_of = 1,
    .ttlt_slo_ms = INT32_MAX,
    .ttft_slo_ms = INT32_MAX,
    .tpot_slo_ms = INT32_MAX,
    .beam_width = 0,
    .num_return_sequences = 0,
    .top_logprobs = 0,
    .top_k = -1,
    .top_p = 1.0,
    .frequency_penalty = 0.0,
    .presence_penalty = 0.0,
    .repetition_penalty = 1.0,
    .temperature = 0.0,
    .request_id = ""};

const XLLM_InitOptions XLLM_INIT_REC_OPTIONS_DEFAULT = {
    .enable_chunked_prefill = false,
    .enable_prefill_sp = false,
    .enable_prefix_cache = false,
    .enable_disagg_pd = false,
    .enable_pd_ooc = false,
    .enable_schedule_overlap = false,
    .enable_shm = false,

    .transfer_listen_port = 26000,
    .nnodes = 1,
    .node_rank = 0,
    .dp_size = 1,
    .ep_size = 1,
    .block_size = 128,
    .max_cache_size = UINT32_MAX,
    .max_tokens_per_batch = 4096,
    .max_seqs_per_batch = 2,
    .max_tokens_per_chunk_for_prefill = 0,
    .num_speculative_tokens = 0,
    .num_request_handling_threads = 4,
    .expert_parallel_degree = 0,
    .server_idx = 0,
    .beam_width = 128,
    .max_decode_rounds = 2,
    .max_token_per_req = 1000,
    .max_memory_utilization = 0.55,
    .flashinfer_workspace_buffer_size = 134217728,
    .request_queue_size = 16,
    .task = "generate",
    .communication_backend = "lccl",
    .instance_role = "DEFAULT",
    .device_ip = "",
    .master_node_addr = "127.0.0.1:18899",
    .xservice_addr = "",
    .instance_name = "",
    .kv_cache_transfer_mode = "PUSH",
    .log_dir = "/export/Logs",
    .draft_model = "",
    .draft_devices = "",
    .enable_graph = false,
    .enable_rec_fast_sampler = true,
    .enable_prefill_piecewise_graph = false,
    .enable_onerec_prefill_acl_graph = false,
    .enable_xattention_one_stage = false,
    .enable_graph_mode_decode_no_padding = false,
    .enable_block_copy_kernel = false,
    .enable_topk_sorted = false,
    .enable_rec_prefill_only = false,
    .enable_constrained_decoding = true,
    .enable_convert_tokens_to_item = true,
    .enable_output_sku_logprobs = false,
    .enable_extended_item_info = false,
    .each_conversion_threshold = 1,
    .total_conversion_threshold = 256,
    .rec_worker_max_concurrency = 1,
    .enable_multistream_perf_mode = false,
    .enable_onerec_multistream_core_split = false,
    .onerec_multistream_core_ratio = 0.5,
    .constrained_decoding_filter_path = "",
    .cpu_affinity = ""};

const XLLM_RequestParams XLLM_REC_REQUEST_PARAMS_DEFAULT = {
    .echo = false,
    .offline = false,
    .logprobs = false,
    .ignore_eos = false,

    .n = 1,
    .max_tokens = 5120,
    .best_of = 1,
    .ttlt_slo_ms = INT32_MAX,
    .ttft_slo_ms = INT32_MAX,
    .tpot_slo_ms = INT32_MAX,
    .beam_width = 128,
    .num_return_sequences = 0,
    .top_logprobs = 0,
    .top_k = -1,
    .top_p = 1.0,
    .frequency_penalty = 0.0,
    .presence_penalty = 0.0,
    .repetition_penalty = 1.0,
    .temperature = 0.0,
    .request_id = ""};

#ifdef __cplusplus
}
#endif

#endif  // XLLM_LLM_DEFAULT_H

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common/macros.h"
#include "common/types.h"

namespace xllm {

class Options {
 public:
  Options() = default;
  ~Options() = default;

  std::string to_string() const;

 private:
  PROPERTY(std::string, model_path);

  PROPERTY(std::string, model_id);

  PROPERTY(std::optional<std::string>, draft_model_path);

  // model backend
  PROPERTY(std::string, backend);

  // max image num per prompt, default 8
  PROPERTY(int32_t, limit_image_per_prompt) = 8;

  // block size, default 128
  PROPERTY(int32_t, block_size) = 128;

  // the maximum cache size in bytes, default is 0 which means cache size is
  // caculated by available memory * max_memory_utilization
  PROPERTY(int64_t, max_cache_size) = 0;

  // maximum memory utilization allowed, default 0.9
  PROPERTY(double, max_memory_utilization) = 0.9;

  PROPERTY(bool, enable_prefix_cache) = true;

  // maximum encoder cache size in MB (0 disables encoder cache)
  PROPERTY(int64_t, max_encoder_cache_size) = 0;

  // active linear-state slots. 0 derives capacity from the KV cache budget.
  PROPERTY(int64_t, max_linear_state_cache_slots) = 0;

  // max tokens num per batch
  PROPERTY(int32_t, max_tokens_per_batch) = 20480;

  // max sequences num per batch
  PROPERTY(int32_t, max_seqs_per_batch) = 256;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill);

  // sps tokens
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "MTP";

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;

  // thread num to handle requests
  PROPERTY(size_t, num_request_handling_threads) = 4;

  PROPERTY(std::optional<bool>, enable_eplb);

  PROPERTY(std::optional<int32_t>, redundant_experts_num);

  PROPERTY(std::optional<int64_t>, eplb_update_interval);

  PROPERTY(std::optional<double>, eplb_update_threshold);

  PROPERTY(std::optional<std::string>, communication_backend);

  PROPERTY(std::optional<std::string>, rank_tablefile);

  PROPERTY(std::optional<int32_t>, expert_parallel_degree);

  PROPERTY(std::string, task_type);

  PROPERTY(bool, enable_mla) = false;

  PROPERTY(std::string, npu_kernel_backend) = "AUTO";

  PROPERTY(bool, enable_chunked_prefill) = true;

  PROPERTY(std::optional<std::string>, master_node_addr);

  PROPERTY(int32_t, nnodes) = 1;

  PROPERTY(int32_t, node_rank) = 0;

  PROPERTY(bool, enable_single_process) = false;

  PROPERTY(int32_t, dp_size) = 1;

  PROPERTY(int32_t, cp_size) = 1;

  PROPERTY(int32_t, ep_size) = 1;

  PROPERTY(int32_t, tp_size) = 1;

  PROPERTY(int32_t, sp_size) = 1;

  PROPERTY(int32_t, cfg_size) = 1;

  PROPERTY(int32_t, vae_size) = 1;

  PROPERTY(std::optional<std::string>, instance_name);

  PROPERTY(bool, enable_disagg_pd) = false;

  PROPERTY(bool, enable_pd_ooc) = false;

  PROPERTY(bool, enable_schedule_overlap) = true;

  PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(uint16_t, transfer_listen_port) = 26000;

  PROPERTY(std::optional<std::string>, etcd_addr);

  PROPERTY(std::optional<std::string>, etcd_namespace);

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(std::optional<std::string>, tool_call_parser);

  PROPERTY(std::optional<std::string>, reasoning_parser);

  PROPERTY(std::string, priority_strategy) = "fcfs";

  PROPERTY(bool, enable_online_preempt_offline) = true;

  PROPERTY(double, host_blocks_factor) = 0.0;

  PROPERTY(bool, enable_kvcache_store) = false;

  PROPERTY(std::string, store_protocol) = "tcp";

  PROPERTY(std::string, store_master_server_address) = "";

  PROPERTY(std::string, store_metadata_server) = "";

  PROPERTY(std::string, store_local_hostname) = "";

  PROPERTY(bool, enable_multi_stream_parallel) = false;

  PROPERTY(bool, enable_profile_step_time) = false;

  PROPERTY(bool, enable_profile_token_budget) = false;

  PROPERTY(bool, enable_latency_aware_schedule) = false;
  // the max prompt length for profile
  PROPERTY(int32_t, profile_max_prompt_length) = 2048;
  // true if generate kv cache for profile
  PROPERTY(bool, enable_profile_kv_blocks) = true;
  // true if disable ttft profiling
  PROPERTY(bool, disable_ttft_profiling) = false;
  // true if enable forward interruption
  PROPERTY(bool, enable_forward_interruption) = false;
  // enable CUDA graph/ACL graph for performance optimization
  PROPERTY(bool, enable_graph) = false;
  // enable graph-mode decode without padding
  PROPERTY(bool, enable_graph_mode_decode_no_padding) = false;
  // enable piecewise graph for prefill
  PROPERTY(bool, enable_prefill_piecewise_graph) = false;
  // maximum number of tokens for graph execution
  PROPERTY(int32_t, max_tokens_for_graph_mode) = 2048;
  // all requests use single global ttft
  PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();
  // all requests use single global tpot
  PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

  // dit
  // max requests per batch
  PROPERTY(int, max_requests_per_batch) = 0;

  // for offline inference: start with offline inference, default is false
  PROPERTY(bool, enable_offline_inference) = false;

  // Enable the RL sleep/wakeup memory mode (SleepableAllocator). When set, the
  // offline Python sleep()/wake_up() interface releases / re-acquires physical
  // HBM via the VMM-backed allocator (vllm-ascend style), independent of the
  // xtensor-based online sleep/wakeup path.
  PROPERTY(bool, enable_sleep_mode) = false;

  // disable per-request statistic logs.
  PROPERTY(bool, disable_log_stats) = false;
  // for offline inference: the path to spawn worker binary
  PROPERTY(std::string, spawn_worker_path) = "";

  // use shared memory for inter-process communication in the single-machine
  // multi-GPU scenario.
  PROPERTY(bool, enable_shm) = false;

  PROPERTY(uint64_t, input_shm_size) = 1024;

  PROPERTY(uint64_t, output_shm_size) = 128;

  // whether the worker and master are on the same machine.
  PROPERTY(bool, is_local) = false;

  // Index ID for internal server ID, which must be set different values
  // if the model supports multiple version or there are multiple models.
  PROPERTY(int64_t, server_idx) = 0;

  // Prefetch timeout for prefetch from kv cache store
  PROPERTY(uint32_t, prefetch_timeout) = 0;

  // Prefetch from kvcache store copy batch size
  PROPERTY(uint32_t, prefetch_batch_size) = 2;

  // Layer wise H2D copy batchs
  PROPERTY(uint32_t, layers_wise_copy_batchs) = 4;

  // beam width for beam search
  PROPERTY(int32_t, beam_width) = 128;

  // KV cache data type for quantization.
  // "auto" (default): KV cache dtype aligns with model dtype (no quantization).
  // "int8": Enables INT8 quantization. Only supported on MLU backend.
  PROPERTY(std::string, kv_cache_dtype) = "auto";

  // max concurrency for rec worker
  PROPERTY(int32_t, rec_worker_max_concurrency) = 1;

  PROPERTY(MasterStatus, master_status) = MasterStatus::WAKEUP;
};

}  // namespace xllm

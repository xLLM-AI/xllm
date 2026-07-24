/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.h"
#include "common/types.h"

namespace xllm {
namespace runtime {

struct Options {
  PROPERTY(std::string, model_path);

  PROPERTY(std::string, model_id);

  PROPERTY(std::optional<std::string>, draft_model_path);

  // model backend
  PROPERTY(std::string, backend);

  // devices for execute model
  PROPERTY(std::vector<torch::Device>, devices);

  // devices for execute draft model
  PROPERTY(std::vector<torch::Device>, draft_devices);

  // the number of slots per block, default 128, value must be multiple of 16
  PROPERTY(int32_t, block_size) = 128;

  // 0 means that cache size is caculated by available memory
  PROPERTY(int64_t, max_cache_size) = 0;

  // maximum memory utilization allowed, default 0.9
  PROPERTY(double, max_memory_utilization) = 0.9;

  // enable prefix cache
  PROPERTY(bool, enable_prefix_cache) = true;

  // maximum encoder cache size in MB (0 disables encoder cache)
  PROPERTY(int64_t, max_encoder_cache_size) = 0;

  // active linear-state slots. 0 derives capacity from the KV cache budget.
  PROPERTY(int64_t, max_linear_state_cache_slots) = 0;

  // number of decoding tokens per sequence
  // in speculative decoding, it is the number of speculative tokens + 1
  PROPERTY(int64_t, num_decoding_tokens) = 1;

  // the number of speculative tokens per step
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "mtp";

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  // enable speculative decode
  PROPERTY(bool, enable_speculative_decode) = false;

  PROPERTY(bool, enable_mtp_draft_body_tp1) = false;

  PROPERTY(bool, is_draft_engine) = false;

  PROPERTY(int32_t, world_size) = 1;

  // task type, support 'generate' and 'embed' currently
  PROPERTY(std::string, task_type) = "generate";

  PROPERTY(bool, enable_mla) = false;

  PROPERTY(std::string, npu_kernel_backend) = "AUTO";

  // master node address when we launch a multi-node task.
  PROPERTY(std::optional<std::string>, master_node_addr);

  // total nodes num
  PROPERTY(int32_t, nnodes) = 1;

  // the node_rank of current worker process at.
  PROPERTY(int32_t, node_rank) = 0;

  // run all local devices inside a single OS process (one worker thread per
  // device) instead of one process per device.
  PROPERTY(bool, enable_single_process) = false;

  // data parallelism size, currently mainly used for MoE model
  // default set as 1 for non-MoE model
  PROPERTY(int32_t, dp_size) = 1;

  // expert parallelism size, currently mainly used for MoE model
  // Default set as 1 for non-MoE model.
  PROPERTY(int32_t, ep_size) = 1;

  // Context parallelism size
  PROPERTY(int32_t, cp_size) = 1;

  // tensor parallelism size
  // Default set as 1
  PROPERTY(int32_t, tp_size) = 1;

  // sequence parallelism size
  // Default set as 1
  PROPERTY(int32_t, sp_size) = 1;

  // classifier-free guidance parallelism size
  // Default set as 1
  PROPERTY(int32_t, cfg_size) = 1;

  // vae patch parallelism size
  // Default set as 1
  PROPERTY(int32_t, vae_size) = 1;

  // enable enable_schedule_overlap to improve runtime execution efficiency.
  PROPERTY(bool, enable_schedule_overlap) = true;

  // enable chunked prefill.
  PROPERTY(bool, enable_chunked_prefill) = true;

  // enable returning aux_hidden_states in graph executor output.
  PROPERTY(bool, enable_graph_aux_hidden_states) = false;

  // the max sequences limit of a batch.
  PROPERTY(int32_t, max_seqs_per_batch) = 256;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill);

  // for master service, current instance name(ID).
  PROPERTY(std::optional<std::string>, instance_name);

  // enable disaggregated prefill-decode mode.
  PROPERTY(bool, enable_disagg_pd) = false;

  // enable online-offline co-location in disaggregated prefill-decode mode.
  PROPERTY(bool, enable_pd_ooc) = false;

  // instance role, support `DEFAULT`, `PREFILL`, `DECODE`, `MIX`
  PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;

  // transfer kv mode in disaggregated prefill and decode execution.
  // support `PUSH` and `PULL`
  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  // transfer_listen_port needed in disaggregated prefill and decode execution.
  PROPERTY(uint16_t, transfer_listen_port) = 26000;

  // enable service routing mode.
  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(std::string, priority_strategy) = "fcfs";

  PROPERTY(bool, enable_online_preempt_offline) = true;

  // host block factor, e.g. host block num = host_blocks_factor * hbm block num
  PROPERTY(double, host_blocks_factor) = 0.0;

  // enable kvcache store.
  PROPERTY(bool, enable_kvcache_store) = false;

  // store transfer protocol.
  PROPERTY(std::string, store_protocol) = "tcp";

  // The address information of the Master (IP:Port for default mode and
  // etcd://IP:Port;IP:Port;...;IP:Port for high availability mode)
  PROPERTY(std::string, store_master_server_address) = "";

  // the address of the metadata service (e.g., etcd/Redis) required for
  // Transfer Engine initialization
  PROPERTY(std::string, store_metadata_server) = "";

  //  the IP:Port of the local machine or an accessible domain name (default
  //  value used if port is not included)
  PROPERTY(std::string, store_local_hostname) = "";

  // Prefetch from kvcache store copy batch size
  PROPERTY(uint32_t, prefetch_batch_size) = 2;

  // Layer wise H2D copy batchs
  PROPERTY(uint32_t, layers_wise_copy_batchs) = 4;

  // dit
  // max requests per batch
  PROPERTY(int, max_requests_per_batch) = 0;

  // start with offline inference, default is false
  PROPERTY(bool, enable_offline_inference) = false;

  // enable RL sleep/wakeup memory mode (SleepableAllocator). Propagated from
  // the user-facing Options so that the engine and worker route the offline
  // sleep()/wake_up() to the VMM-backed allocator instead of xtensor.
  PROPERTY(bool, enable_sleep_mode) = false;

  // disable per-request statistic logs.
  PROPERTY(bool, disable_log_stats) = false;

  // the path to spawn worker binary
  PROPERTY(std::string, spawn_worker_path) = "";

  // use shared memory for inter-process communication in the single-machine
  // multi-GPU scenario.
  PROPERTY(bool, enable_shm) = false;

  // Input shared memory size
  PROPERTY(uint64_t, input_shm_size) = 1024;

  // Output shared memory size
  PROPERTY(uint64_t, output_shm_size) = 128;

  // whether the worker and master are on the same machine.
  PROPERTY(bool, is_local) = false;

  // Index ID for internal server ID, which must be set different values
  // if the model supports multiple version or there are multiple models.
  PROPERTY(int64_t, server_idx) = 0;

  // enable CUDA graph/ACL graph for performance optimization
  PROPERTY(bool, enable_graph) = false;
  // enable graph-mode decode without padding
  PROPERTY(bool, enable_graph_mode_decode_no_padding) = false;
  // enable piecewise graph for prefill
  PROPERTY(bool, enable_prefill_piecewise_graph) = false;
  // maximum number of tokens for graph execution
  PROPERTY(int32_t, max_tokens_for_graph_mode) = 2048;

  // beam width for beam search
  PROPERTY(int32_t, beam_width) = 128;

  // max tokens per batch
  PROPERTY(int32_t, max_tokens_per_batch) = 4096;

  // KV cache data type for quantization.
  // "auto" (default): KV cache dtype aligns with model dtype (no quantization).
  // "int8": Enables INT8 quantization. Only supported on MLU backend.
  PROPERTY(std::string, kv_cache_dtype) = "auto";

  // max concurrency for rec worker
  PROPERTY(int32_t, rec_worker_max_concurrency) = 1;
};

}  // namespace runtime
}  // namespace xllm

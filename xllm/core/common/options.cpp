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

#include "options.h"

namespace xllm {
std::string Options::to_string() const {
  std::stringstream ss;
  ss << "Options: [";
  ss << "model_path: " << model_path()
     << ", draft_model_path: " << draft_model_path().value_or("null")
     << ", backend: " << backend()
     << ", limit_image_per_prompt: " << limit_image_per_prompt()
     << ", block_size: " << block_size()
     << ", max_cache_size: " << max_cache_size()
     << ", max_memory_utilization: " << max_memory_utilization()
     << ", enable_prefix_cache: " << enable_prefix_cache()
     << ", max_encoder_cache_size: " << max_encoder_cache_size()
     << ", max_linear_state_cache_slots: " << max_linear_state_cache_slots()
     << ", max_tokens_per_batch: " << max_tokens_per_batch()
     << ", max_seqs_per_batch: " << max_seqs_per_batch()
     << ", max_tokens_per_chunk_for_prefill: "
     << max_tokens_per_chunk_for_prefill()
     << ", num_speculative_tokens: " << num_speculative_tokens()
     << ", speculative_algorithm: " << speculative_algorithm()
     << ", speculative_suffix_cache_max_depth: "
     << speculative_suffix_cache_max_depth()
     << ", speculative_suffix_max_spec_factor: "
     << speculative_suffix_max_spec_factor()
     << ", speculative_suffix_max_spec_offset: "
     << speculative_suffix_max_spec_offset()
     << ", speculative_suffix_min_token_prob: "
     << speculative_suffix_min_token_prob()
     << ", speculative_suffix_max_cached_requests: "
     << speculative_suffix_max_cached_requests()
     << ", speculative_suffix_use_tree_spec: "
     << speculative_suffix_use_tree_spec()
     << ", enable_mtp_draft_body_tp1: " << enable_mtp_draft_body_tp1()
     << ", num_request_handling_threads: " << num_request_handling_threads()
     << ", communication_backend: " << communication_backend().value_or("null")
     << ", rank_tablefile: " << rank_tablefile().value_or("null")
     << ", expert_parallel_degree: " << expert_parallel_degree().value_or(0)
     << ", task_type: " << task_type() << ", enable_mla: " << enable_mla()
     << ", enable_chunked_prefill: " << enable_chunked_prefill()
     << ", cp_size: " << cp_size()
     << ", master_node_addr: " << master_node_addr().value_or("null")
     << ", instance_role: " << instance_role().to_string()
     << ", transfer_listen_port: " << transfer_listen_port()
     << ", nnodes: " << nnodes() << ", node_rank: " << node_rank()
     << ", enable_schedule_overlap: " << enable_schedule_overlap()
     << ", enable_disagg_pd: " << enable_disagg_pd()
     << ", enable_pd_ooc: " << enable_pd_ooc()
     << ", disable_log_stats: " << disable_log_stats()
     << ", kv_cache_dtype: " << kv_cache_dtype()
     << ", kv_cache_transfer_mode: " << kv_cache_transfer_mode()
     << ", etcd_addr: " << etcd_addr().value_or("null")
     << ", etcd_namespace: " << etcd_namespace().value_or("null")
     << ", enable_service_routing: " << enable_service_routing()
     << ", enable_kvcache_store: " << enable_kvcache_store()
     << ", prefetch_timeout: " << prefetch_timeout()
     << ", prefetch_batch_size: " << prefetch_batch_size()
     << ", layers_wise_copy_batchs: " << layers_wise_copy_batchs()
     << ", store_protocol: " << store_protocol()
     << ", store_master_server_address: " << store_master_server_address()
     << ", store_metadata_server: " << store_metadata_server()
     << ", store_local_hostname: " << store_local_hostname()
     << ", enable_multi_stream_parallel: " << enable_multi_stream_parallel()
     << ", disable_ttft_profiling: " << disable_ttft_profiling()
     << ", enable_forward_interruption: " << enable_forward_interruption()
     << ", enable_graph: " << enable_graph()
     << ", enable_graph_mode_decode_no_padding: "
     << enable_graph_mode_decode_no_padding()
     << ", enable_prefill_piecewise_graph: " << enable_prefill_piecewise_graph()
     << ", max_tokens_for_graph_mode: " << max_tokens_for_graph_mode()
     << ", server_idx: " << server_idx();
  ss << "]";
  return ss.str();
}

}  // namespace xllm

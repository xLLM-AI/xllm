/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/distributed_runtime/runtime_options_builder.h"

namespace xllm {

runtime::Options make_runtime_options(
    const Options& options,
    const std::vector<torch::Device>& devices) {
  runtime::Options runtime_options;
  runtime_options.model_path(options.model_path())
      .model_id(options.model_id())
      .devices(devices)
      .backend(options.backend())
      .block_size(options.block_size())
      .max_cache_size(options.max_cache_size())
      .max_memory_utilization(options.max_memory_utilization())
      .enable_prefix_cache(options.enable_prefix_cache())
      .max_encoder_cache_size(options.max_encoder_cache_size())
      .max_linear_state_cache_slots(options.max_linear_state_cache_slots())
      .task_type(options.task_type())
      .enable_mla(options.enable_mla())
      .npu_kernel_backend(options.npu_kernel_backend())
      .master_node_addr(options.master_node_addr())
      .nnodes(options.nnodes())
      .node_rank(options.node_rank())
      .dp_size(options.dp_size())
      .ep_size(options.ep_size())
      .cp_size(options.cp_size())
      .enable_schedule_overlap(options.enable_schedule_overlap())
      .enable_chunked_prefill(options.enable_chunked_prefill())
      .enable_flashcomm1(options.enable_flashcomm1())
      .flashcomm1_min_prefill_tokens(options.flashcomm1_min_prefill_tokens())
      .enable_mmrs_fusion(options.enable_mmrs_fusion())
      .mmrs_comm_mode(options.mmrs_comm_mode())
      .max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .instance_name(options.instance_name())
      .enable_disagg_pd(options.enable_disagg_pd())
      .enable_pd_ooc(options.enable_pd_ooc())
      .instance_role(options.instance_role())
      .kv_cache_transfer_mode(options.kv_cache_transfer_mode())
      .transfer_listen_port(options.transfer_listen_port())
      .enable_service_routing(options.enable_service_routing())
      .priority_strategy(options.priority_strategy())
      .enable_online_preempt_offline(options.enable_online_preempt_offline())
      .host_blocks_factor(options.host_blocks_factor())
      .enable_kvcache_store(options.enable_kvcache_store())
      .store_protocol(options.store_protocol())
      .store_master_server_address(options.store_master_server_address())
      .store_metadata_server(options.store_metadata_server())
      .store_local_hostname(options.store_local_hostname())
      .prefetch_batch_size(options.prefetch_batch_size())
      .layers_wise_copy_batchs(options.layers_wise_copy_batchs())
      .enable_offline_inference(options.enable_offline_inference())
      .enable_sleep_mode(options.enable_sleep_mode())
      .disable_log_stats(options.disable_log_stats())
      .spawn_worker_path(options.spawn_worker_path())
      .enable_shm(options.enable_shm())
      .input_shm_size(options.input_shm_size() * 1024 * 1024)
      .output_shm_size(options.output_shm_size() * 1024 * 1024)
      .is_local(options.is_local())
      .server_idx(options.server_idx())
      .enable_graph(options.enable_graph())
      .enable_graph_mode_decode_no_padding(
          options.enable_graph_mode_decode_no_padding())
      .enable_prefill_piecewise_graph(options.enable_prefill_piecewise_graph())
      .max_tokens_for_graph_mode(options.max_tokens_for_graph_mode())
      .kv_cache_dtype(options.kv_cache_dtype());
  return runtime_options;
}

runtime::Options make_speculative_runtime_options(
    const Options& options,
    const std::vector<torch::Device>& devices,
    const std::vector<torch::Device>& draft_devices) {
  runtime::Options runtime_options = make_runtime_options(options, devices);
  runtime_options.draft_model_path(options.draft_model_path().value_or(""))
      .draft_devices(draft_devices)
      .num_speculative_tokens(options.num_speculative_tokens())
      .speculative_algorithm(options.speculative_algorithm())
      .enable_mtp_draft_body_tp1(options.enable_mtp_draft_body_tp1())
      .speculative_suffix_cache_max_depth(
          options.speculative_suffix_cache_max_depth())
      .speculative_suffix_max_spec_factor(
          options.speculative_suffix_max_spec_factor())
      .speculative_suffix_max_spec_offset(
          options.speculative_suffix_max_spec_offset())
      .speculative_suffix_min_token_prob(
          options.speculative_suffix_min_token_prob())
      .speculative_suffix_max_cached_requests(
          options.speculative_suffix_max_cached_requests())
      .speculative_suffix_use_tree_spec(
          options.speculative_suffix_use_tree_spec());
  return runtime_options;
}

}  // namespace xllm

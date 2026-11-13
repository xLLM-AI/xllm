/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include "core/framework/model/vlm_model_params.h"
#include "core/runtime/forward_params.h"

namespace xllm {

class VlmForwardInput;
namespace detail {
bool unpack_from_input_host_buffer(const VlmForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   VlmForwardInput& output,
                                   bool materialize_device_buffer);
}  // namespace detail

class VlmForwardInput final {
 public:
  VlmForwardInput() = default;
  VlmForwardInput(const VlmForwardInput&) = delete;
  VlmForwardInput& operator=(const VlmForwardInput&) = delete;
  VlmForwardInput(VlmForwardInput&&) = default;
  VlmForwardInput& operator=(VlmForwardInput&&) = default;

  bool to_packed_proto(proto::PackedForwardInput* packed_forward_input) const;
  bool from_packed_proto(const proto::PackedForwardInput& packed_forward_input);
  // Large continuous CPU tensors are retained by the attachment. Callers must
  // keep their storage immutable until the attachment is released.
  bool to_packed_attachment(proto::PackedForwardInput* metadata,
                            butil::IOBuf& attachment) const;
  bool from_packed_attachment(const proto::PackedForwardInput& metadata,
                              const butil::IOBuf& attachment);

  VlmForwardInput clone() const {
    VlmForwardInput inputs;
    inputs.runtime = runtime;
    copy_non_runtime_metadata_to(inputs);
    inputs.token_ids = token_ids;
    inputs.positions = positions;
    inputs.token_ids_host = token_ids_host;
    inputs.positions_host = positions_host;
    inputs.input_params = input_params.clone();
    inputs.sampling_params = sampling_params;
    inputs.json_object_errors = json_object_errors;
    return inputs;
  }

  VlmForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const;

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  VlmForwardInput& inputs) const;

  void copy_metadata_to(VlmForwardInput& inputs) const {
    copy_non_runtime_metadata_to(inputs);
    inputs.runtime.kv_slot_layout = runtime.kv_slot_layout;
    inputs.runtime.metadata_ready_event = runtime.metadata_ready_event;
    inputs.runtime.retained_device_tensors = runtime.retained_device_tensors;
  }

  void copy_non_runtime_metadata_to(VlmForwardInput& inputs) const {
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.input_host_sample_count = input_host_sample_count;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.sample_sequence_ids = sample_sequence_ids;
    inputs.sample_prior_output_rows = sample_prior_output_rows;
    inputs.json_object_states = json_object_states;
    inputs.json_object_state_snapshots = json_object_state_snapshots;
  }

  void set_host_views(VlmForwardInput& inputs) const {
    inputs.token_ids_host =
        token_ids_host.defined() ? token_ids_host : cpu_view(token_ids);
    inputs.positions_host =
        positions_host.defined() ? positions_host : cpu_view(positions);
  }

  const torch::Tensor& host_token_ids() const {
    return token_ids_host.defined() ? token_ids_host : token_ids;
  }

  const torch::Tensor& host_positions() const {
    return positions_host.defined() ? positions_host : positions;
  }

  static torch::Tensor cpu_view(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.device().is_cpu()) {
      return tensor;
    }
    return torch::Tensor();
  }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  mutable VlmModelParams input_params;
  SamplingParameters sampling_params;
  std::vector<std::string> sample_sequence_ids;
  std::vector<int32_t> sample_prior_output_rows;
  std::vector<JsonObjectGrammarState> json_object_states;
  std::vector<JsonObjectGrammarSnapshot> json_object_state_snapshots;
  // Errors detected while aligning prior overlap output with grammar rows.
  std::vector<JsonObjectOutputError> json_object_errors;

  // step-level decode metadata
  // If true, skip sampler forward and keep raw logits; the speculative
  // consumer applies the sampling transforms itself.
  bool skip_sampling_for_logits_only = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;

  int32_t input_host_sample_count = kUnknownPackedSampleCount;
  ForwardRuntimeState runtime;
};

inline LlmForwardInput make_llm_draft_input(const LlmForwardInput& target,
                                            bool shared_full_cache = false) {
  LlmForwardInput draft = target.clone();
  draft.input_params.clear_linear_attention_state();
  if (!shared_full_cache) {
    draft.input_params.attention.host.shared_full_block_table = torch::Tensor();
    draft.input_params.attention.host.shared_full_slot_mapping =
        torch::Tensor();
    draft.input_params.attention.device.shared_full_block_table =
        torch::Tensor();
    draft.input_params.attention.device.shared_full_slot_mapping =
        torch::Tensor();
  }
  return draft;
}

inline LlmForwardInput make_llm_draft_input(const VlmForwardInput& target,
                                            bool shared_full_cache = false) {
  CHECK(!target.runtime.input_host_buffer_has_layout ||
        target.runtime.device_tensors_ready)
      << "Cannot create an LLM draft input from an unmaterialized packed VLM "
         "input.";
  LlmForwardInput draft;
  draft.token_ids = target.token_ids;
  draft.positions = target.positions;
  draft.token_ids_host = target.token_ids_host;
  draft.positions_host = target.positions_host;
  auto& params = draft.input_params;
  const auto& source = target.input_params;
  params.meta = source.meta;
  detail::copy_attention_host_input(source.attention.host,
                                    params.attention.host);
  detail::copy_attention_device_input(source.attention.device,
                                      params.attention.device);
  if (!shared_full_cache) {
    params.attention.host.shared_full_block_table = torch::Tensor();
    params.attention.host.shared_full_slot_mapping = torch::Tensor();
    params.attention.device.shared_full_block_table = torch::Tensor();
    params.attention.device.shared_full_slot_mapping = torch::Tensor();
  }
  params.attention.attention_host_buffer =
      source.attention.attention_host_buffer;
  params.attention.attention_device_buffer =
      source.attention.attention_device_buffer;
  params.attention.attention_buffer_bytes =
      source.attention.attention_buffer_bytes;
  params.attention.attention_buffer_capacity =
      source.attention.attention_buffer_capacity;
  params.attention.attention_buffer_owner =
      source.attention.attention_buffer_owner;
  params.embedding.input_embedding = source.embedding.input_embedding;
  params.embedding.embedding_ids = source.embedding.embedding_ids;
  params.embedding.request_ids = source.embedding.request_ids;
  params.embedding.extra_token_ids = source.embedding.extra_token_ids;
  params.embedding.mtp_bootstrap_row_idxes =
      source.embedding.mtp_bootstrap_row_idxes;
  params.embedding.mtp_bootstrap_embeddings =
      source.embedding.mtp_bootstrap_embeddings;
  params.parallel = source.parallel;
  params.expert = source.expert;
  params.graph.attn_mask = source.graph.attn_mask;
  params.graph.tiling_data = source.graph.tiling_data;
#if defined(USE_DCU)
  params.graph.use_dense_flash_attention =
      source.graph.use_dense_flash_attention;
#endif
  params.graph.expanded_kv_seq_lens = source.graph.expanded_kv_seq_lens;
  params.graph.expanded_block_tables = source.graph.expanded_block_tables;
  params.graph.expanded_paged_kv_indptr = source.graph.expanded_paged_kv_indptr;
  params.graph.expanded_paged_kv_indices =
      source.graph.expanded_paged_kv_indices;
  params.graph.expanded_paged_kv_last_page_len =
      source.graph.expanded_paged_kv_last_page_len;
  params.graph.expanded_tiling_data = source.graph.expanded_tiling_data;
  params.graph.expanded_kv_seq_lens_vec = source.graph.expanded_kv_seq_lens_vec;
#if defined(USE_NPU)
  params.graph.acl_graph_task_update_context =
      source.graph.acl_graph_task_update_context;
#endif
  params.graph.input_tokens_override = source.graph.input_tokens_override;
  params.graph.spec_verify_draft_token_sources =
      source.graph.spec_verify_draft_token_sources;
  params.graph.spec_verify_source_addresses_stable =
      source.graph.spec_verify_source_addresses_stable;
  params.graph.spec_verify_static_graph_tasks_prepared =
      source.graph.spec_verify_static_graph_tasks_prepared;
  params.multi_block_tables = source.multi_block_tables;
  params.prefill_without_cache = source.prefill_without_cache;
  params.num_accepted_tokens = source.num_accepted_tokens;
  params.mtp_topk_state = source.mtp_topk_state;
  params.num_accepted_tokens_host = source.num_accepted_tokens_host;
  params.attn_metadata = source.attn_metadata;
  params.enable_graph = source.enable_graph;
  params.clear_linear_attention_state();
  draft.sampling_params = target.sampling_params;
  draft.sample_sequence_ids = target.sample_sequence_ids;
  draft.sample_prior_output_rows = target.sample_prior_output_rows;
  draft.json_object_states = target.json_object_states;
  draft.json_object_state_snapshots = target.json_object_state_snapshots;
  draft.skip_sampling_for_logits_only = target.skip_sampling_for_logits_only;
  draft.transfer_kv_infos = target.transfer_kv_infos;
  draft.runtime = target.runtime;
  draft.runtime.input_host_buffer_has_layout = false;
  return draft;
}

}  // namespace xllm

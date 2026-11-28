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

#include "core/runtime/vlm_forward_params.h"

#include "core/framework/config/execution_config.h"

namespace xllm {

VlmForwardInput VlmForwardInput::clone() const {
  VlmForwardInput inputs;
  inputs.runtime = runtime;
  copy_non_runtime_metadata_to(inputs);
  inputs.token_ids = token_ids;
  inputs.positions = positions;
  inputs.token_ids_host = token_ids_host;
  inputs.positions_host = positions_host;
  inputs.input_params = input_params.clone();
  inputs.mm_data_ = mm_data_;
  inputs.sampling_params = sampling_params;
  inputs.json_object_errors = json_object_errors;
  return inputs;
}

VlmForwardInput VlmForwardInput::to(const torch::Device& device,
                                    torch::ScalarType dtype) const {
  if (runtime.device_tensors_ready) {
    VlmForwardInput inputs = clone();
    inputs.runtime.input_host_buffer_has_layout = false;
    return inputs;
  }

  const bool use_contiguous_input_buffer =
      ::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
      detail::supports_contiguous_forward_input_buffer(device);
  if (runtime.input_host_buffer_has_layout) {
    VlmForwardInput buffer_inputs;
    if (detail::unpack_from_input_host_buffer(
            *this, device, dtype, buffer_inputs, use_contiguous_input_buffer)) {
      if (buffer_inputs.runtime.device_tensors_ready) {
        buffer_inputs.runtime.input_host_buffer_has_layout = false;
        return buffer_inputs;
      }
      return buffer_inputs.to(device, dtype);
    }
  }

  if (use_contiguous_input_buffer) {
    VlmForwardInput contiguous_inputs;
    if (to_contiguous_input_buffer(device, contiguous_inputs)) {
      return contiguous_inputs;
    }
  }

  VlmForwardInput inputs;
  set_host_views(inputs);
  const torch::Tensor& source_token_ids =
      inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
  const torch::Tensor& source_positions =
      inputs.positions_host.defined() ? inputs.positions_host : positions;
  inputs.token_ids = safe_to(source_token_ids, device, /*non_blocking=*/true);
  inputs.positions = detail::normalize_positions_for_device(
      safe_to(source_positions, device, /*non_blocking=*/true));
  inputs.input_params = input_params.to(device);
  inputs.mm_data_ = MMBatchData::to(mm_data_, device);
  inputs.sampling_params = sampling_params.to(device, dtype);
  copy_metadata_to(inputs);
  inputs.runtime.input_host_buffer = runtime.input_host_buffer;
  inputs.runtime.device_input_buffer = runtime.device_input_buffer;
  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.kv_slot_layout = runtime.kv_slot_layout;
  return inputs;
}

bool VlmForwardInput::to_contiguous_input_buffer(
    const torch::Device& device,
    VlmForwardInput& inputs) const {
  copy_metadata_to(inputs);
  set_host_views(inputs);

  const VlmModelParams& source_params = input_params;
  if ((token_ids.defined() && !inputs.token_ids_host.defined()) ||
      (positions.defined() && !inputs.positions_host.defined()) ||
      mm_data_.valid() || !source_params.embedding.deep_stacks_.empty()) {
    return false;
  }

  inputs.input_params = source_params.clone();
  detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);
  inputs.sampling_params = sampling_params;

  torch::Tensor positions_for_device =
      detail::normalize_positions_for_device(inputs.positions_host);
  detail::ForwardInputBufferPlan plan;
  if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
      !plan.add(positions_for_device, &inputs.positions) ||
      !detail::add_attention_to_plan(
          source_params.attention, inputs.input_params.attention, plan) ||
      !detail::add_model_tensors_to_plan(
          source_params, inputs.input_params, plan) ||
      !detail::add_sampling_to_plan(
          sampling_params, inputs.sampling_params, plan)) {
    return false;
  }

  const uint64_t total_bytes = plan.prepare_layout();
  if (total_bytes > 0) {
    inputs.runtime.input_host_buffer = plan.build_host_buffer(total_bytes);
    inputs.runtime.device_input_buffer =
        safe_to(inputs.runtime.input_host_buffer,
                torch::TensorOptions().dtype(torch::kUInt8).device(device),
                /*non_blocking=*/true);
    plan.bind_device_views(inputs.runtime.device_input_buffer, device);
  }

  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.input_host_buffer_has_layout = false;
  return true;
}

LlmForwardInput make_llm_draft_input(const VlmForwardInput& target,
                                     bool shared_full_cache) {
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
  params.graph = source.graph;
  params.multi_block_tables = source.multi_block_tables;
  params.prefill_without_cache = source.prefill_without_cache;
  params.num_accepted_tokens = source.num_accepted_tokens;
  params.mtp_topk_state = source.mtp_topk_state;
  params.num_accepted_tokens_host = source.num_accepted_tokens_host;
  params.attn_metadata = source.attn_metadata;
  params.enable_graph = source.enable_graph;
  draft.sampling_params = target.sampling_params;
  draft.sample_sequence_ids = target.sample_sequence_ids;
  draft.sample_prior_output_rows = target.sample_prior_output_rows;
  draft.json_object_states = target.json_object_states;
  draft.json_object_state_snapshots = target.json_object_state_snapshots;
  draft.skip_sampling_for_logits_only = target.skip_sampling_for_logits_only;
  draft.return_selected_hidden = target.return_selected_hidden;
  draft.transfer_kv_infos = target.transfer_kv_infos;
  draft.runtime = target.runtime;
  draft.runtime.input_host_buffer_has_layout = false;
  return draft;
}

}  // namespace xllm

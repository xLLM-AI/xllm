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
#include "core/framework/multimodal/mm_batch_data.h"
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

  VlmForwardInput clone() const;

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
    inputs.return_selected_hidden = return_selected_hidden;
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
  mutable MMBatchData mm_data_;
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
  // If true, populate ForwardOutput.selected_hidden with hidden states matching
  // the `logits` selection layout. Used by DSpark ConfidenceHead which needs
  // pre-lm_head hidden states of the draft tokens.
  bool return_selected_hidden = false;

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

LlmForwardInput make_llm_draft_input(const VlmForwardInput& target,
                                     bool shared_full_cache = false);

}  // namespace xllm

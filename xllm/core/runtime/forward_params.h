/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/types.h>

#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/types.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/constrained/json_object_output_error.h"
#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/multimodal/embedding_output.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/util/tensor_helper.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/dit_forward_params.h"
#include "runtime/forward_runtime_state.h"

namespace butil {
class IOBuf;
}

namespace xllm {

namespace proto {
class PackedForwardInput;
}  // namespace proto

class LlmForwardInput;
constexpr int32_t kUnknownPackedSampleCount = -1;

namespace detail {

inline bool supports_contiguous_forward_input_buffer(
    const torch::Device& device) {
#if defined(USE_CUDA)
  return device.type() == torch::kCUDA;
#elif defined(USE_MLU) || defined(USE_MUSA)
  return device.type() == torch::kPrivateUse1;
#elif defined(USE_NPU)
  (void)device;
  return true;
#else
  (void)device;
  return false;
#endif
}

bool unpack_from_input_host_buffer(const LlmForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   LlmForwardInput& output,
                                   bool materialize_device_buffer);

inline torch::Tensor normalize_positions_for_device(
    const torch::Tensor& positions) {
  if ((Platform::is_cuda() || Platform::is_ilu() || Platform::is_musa()) &&
      positions.defined() && positions.scalar_type() != torch::kInt64) {
    return positions.to(torch::kInt64);
  }
  return positions;
}

}  // namespace detail

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,         // LLM
    VLM,         // VLM
    DIT,         // DIT
    ELM,         // Embedding LM
    EVLM,        // Embedding VLM
    MMEVLM = 7,  // Encoder Embedding VLM
  };

  constexpr WorkerType(Value v) : value_(v) {}
  WorkerType(const std::string& str) {
    if (str == "LLM") {
      value_ = LLM;
    } else if (str == "VLM") {
      value_ = VLM;
    } else if (str == "DIT") {
      value_ = DIT;
    } else if (str == "ELM") {
      value_ = ELM;
    } else if (str == "EVLM") {
      value_ = EVLM;
    } else if (str == "MMEVLM") {
      value_ = MMEVLM;
    } else {
      value_ = INVALID;
    }
  }

  WorkerType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(WorkerType rhs) const { return value_ == rhs.value_; }
  bool operator!=(WorkerType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else if (this->value_ == ELM) {
      return "ELM";
    } else if (this->value_ == EVLM) {
      return "EVLM";
    } else if (this->value_ == MMEVLM) {
      return "MMEVLM";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Inputs for forward execution
class LlmForwardInput final {
 public:
  LlmForwardInput() = default;
  LlmForwardInput(const LlmForwardInput&) = delete;
  LlmForwardInput& operator=(const LlmForwardInput&) = delete;
  LlmForwardInput(LlmForwardInput&&) = default;
  LlmForwardInput& operator=(LlmForwardInput&&) = default;

  bool to_packed_proto(proto::PackedForwardInput* packed_forward_input) const;
  bool from_packed_proto(const proto::PackedForwardInput& packed_forward_input);
  // Large continuous CPU tensors are retained by the attachment. Callers must
  // keep their storage immutable until the attachment is released.
  bool to_packed_attachment(proto::PackedForwardInput* metadata,
                            butil::IOBuf& attachment) const;
  bool from_packed_attachment(const proto::PackedForwardInput& metadata,
                              const butil::IOBuf& attachment);

  LlmForwardInput clone() const {
    LlmForwardInput inputs;
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

  LlmForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const {
    if (runtime.device_tensors_ready) {
      return clone();
    }

    if (runtime.input_host_buffer_has_layout) {
      LlmForwardInput buffer_inputs;
      const bool materialize_device_buffer =
          ::xllm::ExecutionConfig::get_instance()
              .use_contiguous_input_buffer() &&
          detail::supports_contiguous_forward_input_buffer(device);
      if (detail::unpack_from_input_host_buffer(
              *this, device, dtype, buffer_inputs, materialize_device_buffer)) {
        if (buffer_inputs.runtime.device_tensors_ready) {
          return buffer_inputs;
        }
        return buffer_inputs.to(device, dtype);
      }
    }

    if (::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
        detail::supports_contiguous_forward_input_buffer(device)) {
      LlmForwardInput contiguous_inputs;
      if (to_contiguous_input_buffer(device, contiguous_inputs)) {
        return contiguous_inputs;
      }
    }

    LlmForwardInput inputs;
    set_host_views(inputs);
    const torch::Tensor& source_token_ids =
        inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
    const torch::Tensor& source_positions =
        inputs.positions_host.defined() ? inputs.positions_host : positions;
    inputs.token_ids = safe_to(source_token_ids, device, true);
    inputs.positions = detail::normalize_positions_for_device(
        safe_to(source_positions, device, true));
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    copy_metadata_to(inputs);
    inputs.runtime.input_host_buffer = runtime.input_host_buffer;
    inputs.runtime.device_input_buffer = runtime.device_input_buffer;
    inputs.runtime.input_host_buffer_has_layout =
        runtime.input_host_buffer_has_layout;
    inputs.runtime.device_tensors_ready = true;
    return inputs;
  }

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  LlmForwardInput& inputs) const;

  void copy_metadata_to(LlmForwardInput& inputs) const {
    copy_non_runtime_metadata_to(inputs);
    inputs.runtime.metadata_ready_event = runtime.metadata_ready_event;
    inputs.runtime.retained_device_tensors = runtime.retained_device_tensors;
  }

  void copy_non_runtime_metadata_to(LlmForwardInput& inputs) const {
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.input_host_sample_count = input_host_sample_count;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.return_selected_hidden = return_selected_hidden;
    inputs.sample_sequence_ids = sample_sequence_ids;
    inputs.sample_prior_output_rows = sample_prior_output_rows;
    inputs.json_object_states = json_object_states;
    inputs.json_object_state_snapshots = json_object_state_snapshots;
  }

  void set_host_views(LlmForwardInput& inputs) const {
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
  mutable LlmModelParams input_params;
  SamplingParameters sampling_params;
  std::vector<std::string> sample_sequence_ids;
  std::vector<int32_t> sample_prior_output_rows;
  std::vector<JsonObjectGrammarState> json_object_states;
  std::vector<JsonObjectGrammarSnapshot> json_object_state_snapshots;
  // Errors detected while aligning prior overlap output with grammar rows.
  std::vector<JsonObjectOutputError> json_object_errors;

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

// output after forward execution
class ForwardOutput final {
 public:
  // Local runtime handle, not in proto/shm; metrics recording skips warmup.
  bool is_graph_warmup = false;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  // Optional CPU token snapshot, valid after ready_event. Device next_tokens
  // remains available for continuation; serializers can reuse this D2H copy.
  torch::Tensor next_tokens_host;
  // Schedule-overlap tails defer the stats build to the consumer; local
  // runtime handles only, not in proto/shm.
  SpecVerifyStats spec_verify_stats;
  bool is_verify_stats_deferred = false;
  std::vector<char> constrained_mask;
  std::vector<JsonObjectOutputError> json_object_errors;
  std::vector<std::string> failed_request_ids;
  // Keep no-sync input tensor handles alive until downstream consumers finish
  // using outputs on the same compute stream. Composite workers append child
  // outputs' retained inputs here. Local runtime handles; not in proto/shm.
  std::vector<std::shared_ptr<const void>> retained_inputs;
  // Device-side readiness dependency for no-sync outputs. This local runtime
  // handle is intentionally not included in proto or shared-memory transport.
  StreamEventPtr ready_event;
  torch::Tensor logits;
  torch::Tensor embedding;
  // Selected hidden states matching `logits` layout: [num_selected,
  // hidden_dim]. Populated when a speculative worker requests the pre-lm_head
  // hidden (e.g. DSpark's ConfidenceHead), or on the context-parallel
  // speculative-decode path, where output_spec_hidden_states reuses it instead
  // of re-selecting the local shard.
  torch::Tensor selected_hidden;
  // Backend-neutral state for the next MTP draft step.
  MtpTopkStatePtr mtp_topk_state;

  // for eplb, collect the tokens load of experts on each worker.
  torch::Tensor expert_load_data;
  // EPLB prepare-attempt token completed by this worker.
  int64_t prepared_token = -1;

  // dit output data
  DiTForwardOutput dit_forward_output;
};

inline std::optional<ForwardOutput> make_transfer_failure_output(
    std::vector<std::string> failed_request_ids) {
  if (failed_request_ids.empty()) {
    return std::nullopt;
  }
  ForwardOutput output;
  output.failed_request_ids = std::move(failed_request_ids);
  return output;
}

inline void append_failed_request_ids(
    ForwardOutput& destination,
    std::vector<std::string>&& failed_request_ids) {
  if (failed_request_ids.empty()) {
    return;
  }
  if (destination.failed_request_ids.empty()) {
    destination.failed_request_ids = std::move(failed_request_ids);
    return;
  }
  destination.failed_request_ids.insert(
      destination.failed_request_ids.end(),
      std::make_move_iterator(failed_request_ids.begin()),
      std::make_move_iterator(failed_request_ids.end()));
  failed_request_ids.clear();
}

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
  // multimodal embedding output for this sequence
  std::vector<torch::Tensor> mm_embeddings;
  EmbeddingOutputMetadata mm_embeddings_metadata;
  SpeculativeTokenStats speculative_token_stats;
};

class RawForwardOutput final {
 public:
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<JsonObjectOutputError> json_object_errors;
  std::vector<std::string> failed_request_ids;
  std::vector<int64_t> expert_load_data;
  int64_t prepared_token = -1;
  // dit output data
  DiTForwardOutput dit_forward_output;
};

}  // namespace xllm

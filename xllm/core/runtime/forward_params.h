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

#include <algorithm>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/speculative/verify_layout.h"
#include "framework/config/execution_config.h"
#include "framework/model/llm_model_params.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/json_object_grammar.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/dit_forward_params.h"
#include "runtime/forward_runtime_state.h"
#include "runtime/json_object_output_rows.h"
#include "util/tensor_helper.h"

namespace xllm {

class LlmForwardInput;
constexpr int32_t kUnknownPackedSampleCount = -1;

namespace detail {

constexpr uint64_t kForwardInputBufferAlignment = 16;

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

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

bool try_to_device_from_input_host_buffer(const LlmForwardInput& input,
                                          const torch::Device& device,
                                          torch::ScalarType dtype,
                                          LlmForwardInput& output);

bool unpack_from_input_host_buffer(const LlmForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   LlmForwardInput& output,
                                   bool materialize_device_buffer);

bool unpack_from_input_host_buffer(const LlmForwardInput& input,
                                   const torch::Device& device,
                                   LlmForwardInput& output);

struct ForwardInputBufferEntry {
  torch::Tensor host_tensor;
  torch::Tensor* target = nullptr;
  uint64_t offset = 0;
  uint64_t aligned_bytes = 0;
};

struct ForwardInputBufferPlan {
  std::vector<ForwardInputBufferEntry> entries;

  bool add(const torch::Tensor& tensor, torch::Tensor* target) {
    if (!tensor.defined()) {
      return true;
    }
    if (!tensor.device().is_cpu()) {
      return false;
    }
    entries.push_back({tensor.contiguous(), target, 0, 0});
    return true;
  }

  uint64_t prepare_layout() {
    uint64_t total = 0;
    for (auto& entry : entries) {
      total = align_up(total, kForwardInputBufferAlignment);
      entry.offset = total;
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      entry.aligned_bytes = align_up(bytes, kForwardInputBufferAlignment);
      total += entry.aligned_bytes;
    }
    return total;
  }

  torch::Tensor build_host_buffer(uint64_t total_bytes) const {
    auto buffer = torch::empty({static_cast<int64_t>(total_bytes)},
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
    auto* base = static_cast<char*>(buffer.data_ptr());
    for (const auto& entry : entries) {
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      if (bytes == 0) {
        continue;
      }
      std::memcpy(base + entry.offset, entry.host_tensor.data_ptr(), bytes);
      if (entry.aligned_bytes > bytes) {
        std::memset(base + entry.offset + bytes,
                    0,
                    static_cast<size_t>(entry.aligned_bytes - bytes));
      }
    }
    return buffer;
  }

  void bind_device_views(const torch::Tensor& device_buffer,
                         const torch::Device& device) const {
    const char* base = static_cast<const char*>(device_buffer.data_ptr());
    for (const auto& entry : entries) {
      if (entry.target == nullptr || !entry.host_tensor.defined()) {
        continue;
      }
      const void* ptr = base + entry.offset;
#if defined(USE_CUDA) || defined(USE_DCU)
      if (device.type() == torch::kCUDA) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_MLU) || defined(USE_MUSA)
      if (device.type() == torch::kPrivateUse1) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_NPU)
      *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                           entry.host_tensor.scalar_type(),
                                           ptr);
#else
      (void)device;
#endif
    }
  }
};

inline bool add_sampling_to_plan(const SamplingParameters& source,
                                 SamplingParameters& target,
                                 ForwardInputBufferPlan& plan) {
  return plan.add(source.selected_token_idxes, &target.selected_token_idxes) &&
         plan.add(source.frequency_penalties, &target.frequency_penalties) &&
         plan.add(source.presence_penalties, &target.presence_penalties) &&
         plan.add(source.repetition_penalties, &target.repetition_penalties) &&
         plan.add(source.temperatures, &target.temperatures) &&
         plan.add(source.top_p, &target.top_p) &&
         plan.add(source.top_k, &target.top_k) &&
         plan.add(source.unique_token_ids, &target.unique_token_ids) &&
         plan.add(source.unique_token_counts, &target.unique_token_counts) &&
         plan.add(source.unique_token_ids_lens,
                  &target.unique_token_ids_lens) &&
         plan.add(source.sample_idxes, &target.sample_idxes) &&
         plan.add(source.do_sample, &target.do_sample) &&
         plan.add(source.filter_mask, &target.filter_mask) &&
         plan.add(source.filter_bitmask, &target.filter_bitmask) &&
         plan.add(source.acc_logprob, &target.acc_logprob);
}

inline torch::Tensor normalize_positions_for_device(
    const torch::Tensor& positions) {
  if ((Platform::is_cuda() || Platform::is_ilu() || Platform::is_musa()) &&
      positions.defined() && positions.scalar_type() != torch::kInt64) {
    return positions.to(torch::kInt64);
  }
  return positions;
}

template <typename Params>
inline void clear_contiguous_input_buffer_tensor_targets(Params& params) {
  params.embedding.input_embedding = torch::Tensor();
  params.embedding.linear_state_indices = torch::Tensor();
  params.embedding.mtp_bootstrap_embeddings = torch::Tensor();
  params.block_copy.src_block_indices = torch::Tensor();
  params.block_copy.dst_block_indices = torch::Tensor();
  params.block_copy.cum_sum = torch::Tensor();
  params.graph.attn_mask = torch::Tensor();
  params.graph.tiling_data = torch::Tensor();
}

template <typename Attention>
inline bool add_attention_to_plan(const Attention& source,
                                  Attention& target,
                                  ForwardInputBufferPlan& plan) {
  return plan.add(source.device.q_seq_lens, &target.device.q_seq_lens) &&
         plan.add(source.device.kv_seq_lens, &target.device.kv_seq_lens) &&
         plan.add(source.device.q_cu_seq_lens, &target.device.q_cu_seq_lens) &&
         plan.add(source.device.new_cache_slots,
                  &target.device.new_cache_slots) &&
         plan.add(source.device.block_tables, &target.device.block_tables) &&
         plan.add(source.device.paged_kv_indptr,
                  &target.device.paged_kv_indptr) &&
         plan.add(source.device.paged_kv_indices,
                  &target.device.paged_kv_indices) &&
         plan.add(source.device.paged_kv_last_page_len,
                  &target.device.paged_kv_last_page_len) &&
         plan.add(source.device.new_cache_slot_offsets,
                  &target.device.new_cache_slot_offsets) &&
         plan.add(source.device.kv_cache_start_offsets,
                  &target.device.kv_cache_start_offsets) &&
         plan.add(source.device.kv_cache_tokens_nums,
                  &target.device.kv_cache_tokens_nums) &&
         plan.add(source.device.history_compressed_kv,
                  &target.device.history_compressed_kv) &&
         plan.add(source.device.history_k_rope,
                  &target.device.history_k_rope) &&
         plan.add(source.device.ring_cur_seqlen,
                  &target.device.ring_cur_seqlen) &&
         plan.add(source.device.ring_cache_seqlen,
                  &target.device.ring_cache_seqlen);
}

template <typename Params>
inline bool add_model_tensors_to_plan(const Params& source,
                                      Params& target,
                                      ForwardInputBufferPlan& plan) {
  return plan.add(source.embedding.input_embedding,
                  &target.embedding.input_embedding) &&
         plan.add(source.embedding.linear_state_indices,
                  &target.embedding.linear_state_indices) &&
         plan.add(source.embedding.mtp_bootstrap_embeddings,
                  &target.embedding.mtp_bootstrap_embeddings) &&
         plan.add(source.block_copy.src_block_indices,
                  &target.block_copy.src_block_indices) &&
         plan.add(source.block_copy.dst_block_indices,
                  &target.block_copy.dst_block_indices) &&
         plan.add(source.block_copy.cum_sum, &target.block_copy.cum_sum) &&
         plan.add(source.graph.attn_mask, &target.graph.attn_mask) &&
         plan.add(source.graph.tiling_data, &target.graph.tiling_data);
}

inline torch::Tensor gather_tensor_by_indices(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& indices) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor cpu_tensor = to_cpu_contiguous(tensor);
  torch::Tensor gather_indices = make_cpu_tensor(indices);
  if (cpu_tensor.dim() <= 1) {
    return cpu_tensor.index_select(0, gather_indices);
  }
  CHECK_EQ(cpu_tensor.dim(), 2) << "Expected 1-D or 2-D tensor for CP shard";
  return cpu_tensor.index_select(1, gather_indices);
}

inline torch::Tensor gather_tensor_by_indices_on_dim(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& indices,
    int64_t dim) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor cpu_tensor = to_cpu_contiguous(tensor);
  torch::Tensor gather_indices = make_cpu_tensor(indices);
  return cpu_tensor.index_select(dim, gather_indices);
}

}  // namespace detail

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,     // LLM
    VLM,     // VLM
    DIT,     // DIT
    ELM,     // Embedding LM
    EVLM,    // Embedding VLM
    REC,     // Rec
    MMEVLM,  // Encoder Embedding VLM
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
    } else if (str == "REC") {
      value_ = REC;
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
    } else if (this->value_ == REC) {
      return "REC";
    } else if (this->value_ == MMEVLM) {
      return "MMEVLM";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Step-level decode metadata for Rec multi-round (device loop).
// Inputs for forward execution
class LlmForwardInput final {
 public:
  LlmForwardInput() = default;
  LlmForwardInput(const LlmForwardInput&) = delete;
  LlmForwardInput& operator=(const LlmForwardInput&) = delete;
  LlmForwardInput(LlmForwardInput&&) = default;
  LlmForwardInput& operator=(LlmForwardInput&&) = default;

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
    inputs.json_object_invalid_draft = json_object_invalid_draft;
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
    inputs.runtime.kv_slot_layout = runtime.kv_slot_layout;
    return inputs;
  }

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  LlmForwardInput& inputs) const {
    copy_metadata_to(inputs);
    set_host_views(inputs);

    const LlmModelParams& source_params = input_params;
    if (missing_required_host_views(inputs)) {
      return false;
    }

    inputs.input_params = source_params.clone();
    detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);

    inputs.sampling_params = sampling_params;

    torch::Tensor positions_for_device =
        detail::normalize_positions_for_device(inputs.positions_host);

    detail::ForwardInputBufferPlan plan;
    if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
        !plan.add(positions_for_device, &inputs.positions)) {
      return false;
    }

    if (!detail::add_attention_to_plan(
            source_params.attention, inputs.input_params.attention, plan) ||
        !detail::add_model_tensors_to_plan(
            source_params, inputs.input_params, plan)) {
      return false;
    }

    if (!detail::add_sampling_to_plan(
            sampling_params, inputs.sampling_params, plan)) {
      return false;
    }

    const uint64_t total_bytes = plan.prepare_layout();
    if (total_bytes > 0) {
      inputs.runtime.input_host_buffer = plan.build_host_buffer(total_bytes);
      inputs.runtime.device_input_buffer =
          safe_to(inputs.runtime.input_host_buffer,
                  torch::TensorOptions().dtype(torch::kUInt8).device(device),
                  true);
      plan.bind_device_views(inputs.runtime.device_input_buffer, device);
    }

    inputs.runtime.device_tensors_ready = true;
    inputs.runtime.input_host_buffer_has_layout = false;
    return true;
  }

  void copy_metadata_to(LlmForwardInput& inputs) const {
    copy_non_runtime_metadata_to(inputs);
    inputs.runtime.kv_slot_layout = runtime.kv_slot_layout;
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

  bool missing_required_host_views(const LlmForwardInput& inputs) const {
    return (token_ids.defined() && !inputs.token_ids_host.defined()) ||
           (positions.defined() && !inputs.positions_host.defined());
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

  void print() const {
    LOG(INFO) << "  token_ids: " << token_ids << std::endl;
    LOG(INFO) << "  positions: " << positions << std::endl;
    ModelInputParams(input_params).print();
    LOG(INFO) << " params.selected_token_idxes "
              << sampling_params.selected_token_idxes;
    LOG(INFO) << " params.sample_idxes " << sampling_params.sample_idxes;
    LOG(INFO) << " params.do_sample " << sampling_params.do_sample;
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
  // Flattened [sequence][draft position] flags produced during MTP
  // validation. This is execution-local metadata and is not transported.
  std::vector<uint8_t> json_object_invalid_draft;
  // Errors detected while aligning prior overlap output with grammar rows.
  std::vector<JsonObjectOutputError> json_object_errors;

  // If true, skip sampler forward and only keep logits.
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
struct ForwardOutput {
  // Local runtime handle, not in proto/shm; metrics recording skips warmup.
  bool is_graph_warmup = false;
  // sample parameters for speculative decoding
  torch::Tensor do_sample;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  // Optional CPU token snapshot, valid after ready_event. Device next_tokens
  // remains available for continuation; serializers can reuse this D2H copy.
  torch::Tensor next_tokens_host;
  // Present only for speculative verification: per-row draft widths and
  // JSON-constraint classification, request-ordered. A non-empty layout is
  // the sole marker of a verify result (ordinary decode emits 1-D tokens);
  // consumed before proto/shared-memory serialization.
  std::vector<VerifyRowLayout> spec_verify_layouts;
  // The target sampler applies packed token masks in-place before returning
  // sampled tokens. MTP validation uses this local contract to avoid applying
  // the same mask to target logits a second time.
  bool filter_bitmask_applied_to_logits = false;
  std::vector<JsonObjectOutputError> json_object_errors;
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

  BeamSearchOutput beam_search_output;
  torch::Tensor beam_sequence_group;

  // dit output data
  DiTForwardOutput dit_forward_output;
};

inline void copy_retained_inputs(ForwardOutput& destination,
                                 const ForwardOutput& source) {
  destination.retained_inputs.insert(destination.retained_inputs.end(),
                                     source.retained_inputs.begin(),
                                     source.retained_inputs.end());
}

inline void transfer_retained_inputs(ForwardOutput& destination,
                                     ForwardOutput& source) {
  CHECK_NE(&destination, &source)
      << "transfer_retained_inputs cannot alias source and destination";
  destination.retained_inputs.insert(
      destination.retained_inputs.end(),
      std::make_move_iterator(source.retained_inputs.begin()),
      std::make_move_iterator(source.retained_inputs.end()));
  source.retained_inputs.clear();
}

inline std::vector<std::shared_ptr<const void>> take_retained_inputs(
    ForwardOutput& source) {
  return std::exchange(source.retained_inputs, {});
}

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
  // multimodal embedding output for this sequence
  std::vector<torch::Tensor> mm_embeddings;
  SpeculativeTokenStats speculative_token_stats;
};

struct RawForwardOutput {
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<JsonObjectOutputError> json_object_errors;
  std::vector<int64_t> expert_load_data;
  int64_t prepared_token = -1;
  // beam search kernel output
  std::vector<int32_t> src_seq_idxes;
  std::vector<int32_t> out_tokens;
  std::vector<float> out_logprobs;

  // batch-level beam output for Rec multi-round mode
  std::vector<int32_t> beam_sequence_group;  // flattened 2D
  // dit output data
  DiTForwardOutput dit_forward_output;
};

struct BatchedForwardInputs {
  std::vector<LlmForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
};

}  // namespace xllm

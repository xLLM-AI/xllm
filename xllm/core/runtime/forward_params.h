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

#include <torch/torch.h>

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
#include "framework/config/execution_config.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/dit_forward_params.h"

namespace xllm {

struct ForwardInput;

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
#elif defined(USE_MLU)
  return device.type() == torch::kPrivateUse1;
#elif defined(USE_NPU)
  (void)device;
  return true;
#else
  (void)device;
  return false;
#endif
}

bool try_to_device_from_input_host_buffer(const ForwardInput& input,
                                          const torch::Device& device,
                                          torch::ScalarType dtype,
                                          ForwardInput& output);

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   ForwardInput& output,
                                   bool materialize_device_buffer);

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   ForwardInput& output);

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
#if defined(USE_MLU)
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

inline bool has_contiguous_input_buffer_exclusions(
    const ModelInputParams& params) {
  return params.multimodal.mm_data.valid() || params.has_onerec_params() ||
         params.has_llmrec_params() || params.dit_forward_input.valid() ||
         !params.multimodal.deep_stacks.empty();
}

inline void clear_contiguous_input_buffer_tensor_targets(
    ModelInputParams& params) {
  params.embedding.input_embedding = torch::Tensor();
  params.embedding.linear_state_indices = torch::Tensor();
  params.embedding.mtp_bootstrap_embeddings = torch::Tensor();
  params.block_copy.src_block_indices = torch::Tensor();
  params.block_copy.dst_block_indices = torch::Tensor();
  params.block_copy.cum_sum = torch::Tensor();
  params.graph.attn_mask = torch::Tensor();
  params.graph.tiling_data = torch::Tensor();
}

inline bool add_attention_to_plan(const AttentionInput& source,
                                  AttentionInput& target,
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

inline bool add_model_tensors_to_plan(const ModelInputParams& source,
                                      ModelInputParams& target,
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
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  torch::Tensor gather_indices = torch::tensor(
      indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  if (cpu_tensor.dim() <= 1) {
    return cpu_tensor.index_select(0, gather_indices);
  }
  CHECK_EQ(cpu_tensor.dim(), 2)
      << "Expected 1-D or 2-D tensor for CP partition";
  return cpu_tensor.index_select(1, gather_indices);
}

inline torch::Tensor gather_tensor_by_indices_on_dim(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& indices,
    int64_t dim) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  torch::Tensor gather_indices = torch::tensor(
      indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  return cpu_tensor.index_select(dim, gather_indices);
}

inline torch::Tensor int_vector_to_cpu_tensor(
    const std::vector<int32_t>& values) {
  if (values.empty()) {
    return torch::Tensor();
  }
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

template <typename T>
inline std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  if (cpu_tensor.scalar_type() != get_scalar_type<T>()) {
    cpu_tensor = cpu_tensor.to(get_scalar_type<T>());
  }
  const T* data_ptr = cpu_tensor.data_ptr<T>();
  const size_t size = static_cast<size_t>(cpu_tensor.numel());
  return std::vector<T>(data_ptr, data_ptr + size);
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

// KV slot layout for the NPU model-side CP closure. Worker-local only: it is
// never transported across RPC/proto/shared-memory boundaries.
//   LOGICAL_REAL              - new_cache_slots carries one entry per real
//                               token in global-real order (BlockManager
//                               logical slot space). This is the layout the
//                               BatchInputBuilder produces and the only layout
//                               that may be fed into localize_slots_recovered()
//                               + recompute_new_cache_slots().
//   NPU_CP_RECOVERED_PHYSICAL  - new_cache_slots has already been expanded to
//                               cp_size * local_padded rows in
//                               cp_kv_recover_idx order (real rows carry their
//                               global slot id, virtual-pad rows carry -1) and
//                               remapped to the KV-split layout. Re-entering
//                               worker prepare must skip the expand/remap to
//                               avoid double remap.
enum class KvSlotLayout : int8_t {
  LOGICAL_REAL = 0,
  NPU_CP_RECOVERED_PHYSICAL = 1,
};

// Step-level decode metadata for Rec multi-round (device loop).
struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  // Planned decode kv cache shape: [batch_size * beam_width, n_kv_heads,
  // step_rounds, head_dim]
  std::vector<int64_t> full_kv_shape;
  // Flattened decode positions for each sequence.
  std::vector<int32_t> decode_positions_vec;
};

// Inputs for forward execution
struct ForwardInput {
  ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    if (device_tensors_ready) {
      return *this;
    }

    if (input_host_buffer_has_layout) {
      ForwardInput buffer_inputs;
      const bool materialize_device_buffer =
          ::xllm::ExecutionConfig::get_instance()
              .use_contiguous_input_buffer() &&
          detail::supports_contiguous_forward_input_buffer(device);
      if (detail::unpack_from_input_host_buffer(
              *this, device, dtype, buffer_inputs, materialize_device_buffer)) {
        if (buffer_inputs.device_tensors_ready) {
          return buffer_inputs;
        }
        return buffer_inputs.to(device, dtype);
      }
    }

    if (::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
        detail::supports_contiguous_forward_input_buffer(device)) {
      ForwardInput contiguous_inputs;
      if (to_contiguous_input_buffer(device, contiguous_inputs)) {
        return contiguous_inputs;
      }
    }

    ForwardInput inputs;
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
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    copy_metadata_to(inputs);
    inputs.input_host_buffer = input_host_buffer;
    inputs.device_input_buffer = device_input_buffer;
    inputs.input_host_buffer_has_layout = input_host_buffer_has_layout;
    inputs.device_tensors_ready = true;
    inputs.cp_partitioned = cp_partitioned;
    inputs.kv_slot_layout = kv_slot_layout;
    return inputs;
  }

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  ForwardInput& inputs) const {
    copy_metadata_to(inputs);
    set_host_views(inputs);

    const ModelInputParams& source_params = input_params;
    if (missing_required_host_views(inputs) ||
        detail::has_contiguous_input_buffer_exclusions(source_params)) {
      return false;
    }

    inputs.input_params = source_params;
    detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);

    inputs.sampling_params = sampling_params;
    inputs.decoder_sampling_params = decoder_sampling_params;

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
            sampling_params, inputs.sampling_params, plan) ||
        !detail::add_sampling_to_plan(
            decoder_sampling_params, inputs.decoder_sampling_params, plan)) {
      return false;
    }

    const uint64_t total_bytes = plan.prepare_layout();
    if (total_bytes > 0) {
      inputs.input_host_buffer = plan.build_host_buffer(total_bytes);
      inputs.device_input_buffer =
          safe_to(inputs.input_host_buffer,
                  torch::TensorOptions().dtype(torch::kUInt8).device(device),
                  true);
      plan.bind_device_views(inputs.device_input_buffer, device);
    }

    inputs.device_tensors_ready = true;
    inputs.input_host_buffer_has_layout = false;
    return true;
  }

  void copy_metadata_to(ForwardInput& inputs) const {
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.step_decode = step_decode;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.cp_partitioned = cp_partitioned;
    inputs.kv_slot_layout = kv_slot_layout;
  }

  void set_host_views(ForwardInput& inputs) const {
    inputs.token_ids_host =
        token_ids_host.defined() ? token_ids_host : cpu_view(token_ids);
    inputs.positions_host =
        positions_host.defined() ? positions_host : cpu_view(positions);
  }

  bool missing_required_host_views(const ForwardInput& inputs) const {
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
    input_params.print();
    LOG(INFO) << " params.selected_token_idxes "
              << sampling_params.selected_token_idxes;
    LOG(INFO) << " params.sample_idxes " << sampling_params.sample_idxes;
    LOG(INFO) << " params.do_sample " << sampling_params.do_sample;
  }

  const StepDecodeMeta* step_meta() const {
    return step_decode ? &(*step_decode) : nullptr;
  }

  bool has_step_meta() const { return step_decode.has_value(); }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;

  // step-level decode metadata
  std::optional<StepDecodeMeta> step_decode;
  // If true, skip sampler forward and only keep logits.
  bool skip_sampling_for_logits_only = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;

  // A tensor used to store all device-side input data, with other input tensors
  // constructed based on the address and offset of this tensor.
  torch::Tensor input_host_buffer;
  torch::Tensor device_input_buffer;
  bool input_host_buffer_has_layout = false;

  // True when token_ids, positions, model input tensors and sampling tensors
  // already point to the device-side views for execution. Worker prepare can
  // then skip rebuilding/H2D in ForwardInput::to().
  bool device_tensors_ready = false;

  // True once the worker has produced the per-CP-rank slice (legacy path) or
  // handed the global stream to the model for model-side CP localization. The
  // model-side path keeps this false and uses `parallel.cp_plan` instead.
  bool cp_partitioned = false;

  // Layout of `attention.device.new_cache_slots` for the NPU model-side CP
  // closure. See `KvSlotLayout`. Defaults to LOGICAL_REAL; the worker flips it
  // to NPU_CP_RECOVERED_PHYSICAL after the one-shot expand/remap so a re-entry
  // into worker prepare (e.g. MTP leaf run_llm_no_sync_impl on an already
  // prepared input) does not convert twice.
  KvSlotLayout kv_slot_layout = KvSlotLayout::LOGICAL_REAL;

  // Device-side readiness dependencies for inputs prepared on a different
  // stream. These are local runtime handles and are intentionally not included
  // in proto or shared-memory transport.
  StreamEventPtr metadata_ready_event;
};

// output after forward execution
struct ForwardOutput {
  // sample parameters for speculative decoding
  torch::Tensor do_sample;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  // Keep no-sync input tensor handles alive until downstream consumers finish
  // using outputs on the same compute stream.
  std::shared_ptr<ForwardInput> retained_input;
  // Composite workers retain nested no-sync inputs until their final output is
  // synchronized or its ready event is consumed.
  std::vector<std::shared_ptr<ForwardInput>> retained_input_dependencies;
  // Device-side readiness dependency for no-sync outputs. This local runtime
  // handle is intentionally not included in proto or shared-memory transport.
  StreamEventPtr ready_event;
  torch::Tensor logits;
  torch::Tensor embedding;
  // Backend-neutral state for the next MTP draft step.
  MtpTopkStatePtr mtp_topk_state;

  // for eplb, collect the tokens load of experts on each worker.
  torch::Tensor expert_load_data;
  // for eplb, indicates that the specified layer on the worker
  // has completed the asynchronous loading of new weight.
  int32_t prepared_layer_id;

  BeamSearchOutput beam_search_output;
  torch::Tensor beam_sequence_group;

  // dit output data
  DiTForwardOutput dit_forward_output;
};

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
  // multimodal embedding output for this sequence
  std::vector<torch::Tensor> mm_embeddings;
};

struct RawForwardOutput {
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<int64_t> expert_load_data;
  int32_t prepared_layer_id;
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
  std::vector<ForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
};

}  // namespace xllm

/* Copyright 2026 The xLLM Authors.
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

#include "core/runtime/forward_params.h"

#include <cstring>

#include "core/runtime/vlm_forward_params.h"
#include "core/util/utils.h"

namespace xllm {
namespace {

constexpr int64_t kForwardInputBufferAlignment = 16;

struct ForwardInputTensorBinding {
  torch::Tensor host_tensor;
  torch::Tensor* target = nullptr;
  int64_t offset = 0;
  int64_t size_bytes = 0;
  int64_t aligned_bytes = 0;
};

class ForwardInputBufferBuilder final {
 public:
  bool add(const torch::Tensor& tensor, torch::Tensor* target) {
    if (!tensor.defined()) {
      return true;
    }
    if (!tensor.device().is_cpu()) {
      return false;
    }
    const int64_t size_bytes = static_cast<int64_t>(tensor.nbytes());
    const int64_t aligned_bytes =
        util::align_up(size_bytes, kForwardInputBufferAlignment);
    tensor_bindings_.push_back({tensor.contiguous(),
                                target,
                                total_size_bytes_,
                                size_bytes,
                                aligned_bytes});
    total_size_bytes_ += aligned_bytes;
    return true;
  }

  void copy_to_device(const torch::Device& device,
                      ForwardRuntimeState& runtime) {
    if (total_size_bytes_ == 0) {
      for (const auto& binding : tensor_bindings_) {
        *binding.target = torch::empty(
            binding.host_tensor.sizes(),
            binding.host_tensor.options().device(device).pinned_memory(false));
      }
      return;
    }

    runtime.input_host_buffer = build_host_buffer();
    runtime.device_input_buffer =
        safe_to(runtime.input_host_buffer,
                torch::TensorOptions().dtype(torch::kUInt8).device(device),
                true);
    bind_device_views(runtime.device_input_buffer);
  }

 private:
  torch::Tensor build_host_buffer() const {
    auto buffer = torch::empty({total_size_bytes_},
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
    auto* base = static_cast<char*>(buffer.data_ptr());
    for (const auto& binding : tensor_bindings_) {
      if (binding.size_bytes == 0) {
        continue;
      }
      const size_t bytes = static_cast<size_t>(binding.size_bytes);
      char* dst = base + binding.offset;
      std::memcpy(dst, binding.host_tensor.data_ptr(), bytes);
      const size_t padding =
          static_cast<size_t>(binding.aligned_bytes - binding.size_bytes);
      if (padding != 0) {
        std::memset(dst + bytes, 0, padding);
      }
    }
    return buffer;
  }

  void bind_device_views(const torch::Tensor& device_buffer) const {
    const char* base = static_cast<const char*>(device_buffer.data_ptr());
    [[maybe_unused]] const torch::DeviceType device_type =
        device_buffer.device().type();
    for (const auto& binding : tensor_bindings_) {
      const void* ptr = base + binding.offset;
#if defined(USE_CUDA) || defined(USE_DCU)
      if (device_type == torch::kCUDA) {
        *binding.target =
            get_tensor_from_blob(binding.host_tensor.sizes().vec(),
                                 binding.host_tensor.scalar_type(),
                                 ptr,
                                 device_buffer);
        continue;
      }
#endif
#if defined(USE_MLU) || defined(USE_MUSA)
      if (device_type == torch::kPrivateUse1) {
        *binding.target =
            get_tensor_from_blob(binding.host_tensor.sizes().vec(),
                                 binding.host_tensor.scalar_type(),
                                 ptr,
                                 device_buffer);
        continue;
      }
#endif
#if defined(USE_NPU)
      *binding.target = get_tensor_from_blob(binding.host_tensor.sizes().vec(),
                                             binding.host_tensor.scalar_type(),
                                             ptr);
#endif
    }
  }

  std::vector<ForwardInputTensorBinding> tensor_bindings_;
  int64_t total_size_bytes_ = 0;
};

bool add_sampling_to_buffer(const SamplingParameters& source,
                            SamplingParameters& target,
                            ForwardInputBufferBuilder& buffer_builder) {
  return buffer_builder.add(source.logits_indices(),
                            &target.logits_indices()) &&
         buffer_builder.add(source.frequency_penalties(),
                            &target.frequency_penalties()) &&
         buffer_builder.add(source.presence_penalties(),
                            &target.presence_penalties()) &&
         buffer_builder.add(source.repetition_penalties(),
                            &target.repetition_penalties()) &&
         buffer_builder.add(source.temperatures(), &target.temperatures()) &&
         buffer_builder.add(source.top_p(), &target.top_p()) &&
         buffer_builder.add(source.min_p(), &target.min_p()) &&
         buffer_builder.add(source.logits_bias(), &target.logits_bias()) &&
         buffer_builder.add(source.min_token_positions(),
                            &target.min_token_positions()) &&
         buffer_builder.add(source.min_tokens_remaining(),
                            &target.min_tokens_remaining()) &&
         buffer_builder.add(source.min_token_ids(), &target.min_token_ids()) &&
         buffer_builder.add(source.seeds(), &target.seeds()) &&
         buffer_builder.add(source.seed_offsets(), &target.seed_offsets()) &&
         buffer_builder.add(source.bad_words(), &target.bad_words()) &&
         buffer_builder.add(source.bad_word_suffix(),
                            &target.bad_word_suffix()) &&
         buffer_builder.add(source.prefix_tokens(), &target.prefix_tokens()) &&
         buffer_builder.add(source.top_k(), &target.top_k()) &&
         buffer_builder.add(source.unique_token_ids(),
                            &target.unique_token_ids()) &&
         buffer_builder.add(source.unique_token_counts(),
                            &target.unique_token_counts()) &&
         buffer_builder.add(source.unique_token_ids_lens(),
                            &target.unique_token_ids_lens()) &&
         buffer_builder.add(source.do_sample(), &target.do_sample()) &&
         buffer_builder.add(source.filter_bitmask(), &target.filter_bitmask());
}

template <typename Params>
void clear_contiguous_input_buffer_tensor_targets(Params& params) {
  params.embedding.input_embedding = torch::Tensor();
  params.embedding.linear_state_indices = torch::Tensor();
  params.embedding.mtp_bootstrap_embeddings = torch::Tensor();
  params.graph.attn_mask = torch::Tensor();
  params.graph.tiling_data = torch::Tensor();
}

template <typename Attention>
bool add_attention_to_buffer(const Attention& source,
                             Attention& target,
                             ForwardInputBufferBuilder& buffer_builder) {
  return buffer_builder.add(source.device.q_seq_lens,
                            &target.device.q_seq_lens) &&
         buffer_builder.add(source.device.kv_seq_lens,
                            &target.device.kv_seq_lens) &&
         buffer_builder.add(source.device.q_cu_seq_lens,
                            &target.device.q_cu_seq_lens) &&
         buffer_builder.add(source.device.new_cache_slots,
                            &target.device.new_cache_slots) &&
         buffer_builder.add(source.device.block_tables,
                            &target.device.block_tables) &&
         buffer_builder.add(source.device.shared_full_block_table,
                            &target.device.shared_full_block_table) &&
         buffer_builder.add(source.device.shared_full_slot_mapping,
                            &target.device.shared_full_slot_mapping) &&
         buffer_builder.add(source.device.paged_kv_indptr,
                            &target.device.paged_kv_indptr) &&
         buffer_builder.add(source.device.paged_kv_indices,
                            &target.device.paged_kv_indices) &&
         buffer_builder.add(source.device.paged_kv_last_page_len,
                            &target.device.paged_kv_last_page_len) &&
         buffer_builder.add(source.device.new_cache_slot_offsets,
                            &target.device.new_cache_slot_offsets) &&
         buffer_builder.add(source.device.kv_cache_start_offsets,
                            &target.device.kv_cache_start_offsets) &&
         buffer_builder.add(source.device.kv_cache_tokens_nums,
                            &target.device.kv_cache_tokens_nums) &&
         buffer_builder.add(source.device.history_compressed_kv,
                            &target.device.history_compressed_kv) &&
         buffer_builder.add(source.device.history_k_rope,
                            &target.device.history_k_rope) &&
         buffer_builder.add(source.device.ring_cur_seqlen,
                            &target.device.ring_cur_seqlen) &&
         buffer_builder.add(source.device.ring_cache_seqlen,
                            &target.device.ring_cache_seqlen);
}

template <typename Params>
bool add_model_tensors_to_buffer(const Params& source,
                                 Params& target,
                                 ForwardInputBufferBuilder& buffer_builder) {
  return buffer_builder.add(source.embedding.input_embedding,
                            &target.embedding.input_embedding) &&
         buffer_builder.add(source.embedding.linear_state_indices,
                            &target.embedding.linear_state_indices) &&
         buffer_builder.add(source.embedding.mtp_bootstrap_embeddings,
                            &target.embedding.mtp_bootstrap_embeddings) &&
         buffer_builder.add(source.graph.attn_mask, &target.graph.attn_mask) &&
         buffer_builder.add(source.graph.tiling_data,
                            &target.graph.tiling_data);
}

template <typename Input>
bool pack_inputs_for_device(const Input& source,
                            const torch::Device& device,
                            Input& inputs) {
  if ((source.token_ids.defined() && !inputs.token_ids_host.defined()) ||
      (source.positions.defined() && !inputs.positions_host.defined())) {
    return false;
  }
  const auto& source_params = source.input_params;
  inputs.input_params = source_params.clone();
  clear_contiguous_input_buffer_tensor_targets(inputs.input_params);
  inputs.sampling_params = source.sampling_params;

  ForwardInputBufferBuilder buffer_builder;
  if (!buffer_builder.add(inputs.token_ids_host, &inputs.token_ids) ||
      !buffer_builder.add(
          detail::normalize_positions_for_device(inputs.positions_host),
          &inputs.positions) ||
      !add_attention_to_buffer(source_params.attention,
                               inputs.input_params.attention,
                               buffer_builder) ||
      !add_model_tensors_to_buffer(
          source_params, inputs.input_params, buffer_builder) ||
      !add_sampling_to_buffer(
          source.sampling_params, inputs.sampling_params, buffer_builder)) {
    return false;
  }

  buffer_builder.copy_to_device(device, inputs.runtime);
  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.input_host_buffer_has_layout = false;
  return true;
}

}  // namespace

bool LlmForwardInput::to_contiguous_input_buffer(
    const torch::Device& device,
    LlmForwardInput& inputs) const {
  copy_metadata_to(inputs);
  set_host_views(inputs);
  return pack_inputs_for_device(*this, device, inputs);
}

bool VlmForwardInput::to_contiguous_input_buffer(
    const torch::Device& device,
    VlmForwardInput& inputs) const {
  copy_metadata_to(inputs);
  set_host_views(inputs);
  return !input_params.multimodal.mm_data.valid() &&
         input_params.multimodal.deep_stacks.empty() &&
         pack_inputs_for_device(*this, device, inputs);
}

}  // namespace xllm

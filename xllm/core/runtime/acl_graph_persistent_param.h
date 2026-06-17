/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <acl/acl.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/runtime/options.h"

// Forward declarations for ATB
namespace atb {
class Context;
class Operation;
namespace customize {
struct TilingBufferInfo;
}
}  // namespace atb

namespace xllm::npu {

// Helper class to hold persistent parameters for graph execution
// Multiple AclGraph instances can share the same GraphPersistentParam object
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       bool need_update_attn_mask = false);

  ~GraphPersistentParam();

  // Update persistent tensors with new input data
  // If return_capture_params is true, returns a ModelInputParams with
  // persistent buffer references. padded_num_tokens must be > 0 when
  // return_capture_params is true, used for build new ModelInputParams for
  // capture. If return_capture_params is false, only updates persistent buffers
  // and returns std::nullopt.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens,
                                         bool return_capture_params = false);

  // Getter methods for persistent tensors
  torch::Tensor persistent_tokens(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      return persistent_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_tokens_;
  }
  torch::Tensor persistent_positions(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      int32_t slice_dim = use_mrope_ ? 1 : 0;
      return persistent_positions_
          .slice(
              /*dim=*/slice_dim, /*start=*/0, /*end=*/actual_tokens)
          .contiguous();
    }
    return persistent_positions_;
  }
  torch::Tensor persistent_new_cache_slots(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      return persistent_new_cache_slots_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_new_cache_slots_;
  }
  torch::Tensor persistent_block_tables(uint32_t actual_batch_size = 0) const {
    if (actual_batch_size > 0) {
      return persistent_block_tables_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_block_tables_;
  }
  torch::Tensor persistent_mask(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      return persistent_mask_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_mask_;
  }
  const torch::Tensor& tiling_data() const { return tiling_data_; }
  torch::Tensor hidden_states(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      return hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return hidden_states_;
  }
  // Setter for hidden_states (for assignment)
  void set_hidden_states(const torch::Tensor& value) {
    const uint32_t result_tokens = value.size(0);
    hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens)
        .copy_(value, /*non_blocking=*/true);
  }
  torch::Tensor q_seq_lens(uint32_t actual_batch_size = 0) const {
    if (actual_batch_size > 0) {
      return q_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return q_seq_lens_;
  }
  torch::Tensor kv_seq_lens(uint32_t actual_batch_size = 0) const {
    if (actual_batch_size > 0) {
      return kv_seq_lens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return kv_seq_lens_;
  }
  bool need_update_attn_mask() const { return need_update_attn_mask_; }
  void set_need_update_attn_mask(bool value) { need_update_attn_mask_ = value; }
  bool need_update_attention_plan() const {
    return need_update_attention_plan_;
  }
  torch::Tensor persistent_embedding(uint32_t actual_tokens = 0) const {
    if (actual_tokens > 0) {
      return persistent_embedding_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return persistent_embedding_;
  }
  torch::Tensor persistent_linear_state_indices(
      uint32_t actual_batch_size = 0) const {
    if (actual_batch_size > 0) {
      return persistent_linear_state_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_linear_state_indices_;
  }
  torch::Tensor persistent_num_accepted_tokens(
      uint32_t actual_batch_size = 0) const {
    if (actual_batch_size > 0) {
      return persistent_num_accepted_tokens_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return persistent_num_accepted_tokens_;
  }
  torch::Tensor aux_hidden_states(uint32_t actual_tokens = 0) const {
    if (!aux_hidden_states_.defined() || aux_hidden_states_.numel() == 0) {
      return aux_hidden_states_;
    }
    if (actual_tokens > 0) {
      return aux_hidden_states_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_tokens);
    }
    return aux_hidden_states_;
  }
  // Setter for aux_hidden_states (for assignment)
  void set_aux_hidden_states(const torch::Tensor& value);

 private:
  bool uses_paged_attention_tiling() const {
    return need_update_attention_plan_ && tiling_data_.defined() &&
           tiling_data_.numel() > 0;
  }

  // Initialize ATB context and custom paged attention operation.
  void initialize_paged_attention_plan_context(const torch::Device& device);

  // Update attention mask efficiently from input parameters
  void update_attention_mask(const ModelInputParams& input_params);

  // Update paged attention tiling based on input parameters
  void plan_paged_attention_tiling(const torch::Tensor& tokens,
                                   const torch::Tensor& k_cache,
                                   const torch::Tensor& v_cache,
                                   const torch::Tensor& block_tables,
                                   const ModelInputParams& input_params,
                                   aclrtStream stream);

  std::vector<int32_t> update_expanded_spec_decode_attention(
      const ModelInputParams& input_params,
      uint32_t actual_num_tokens,
      uint32_t padded_num_tokens,
      int64_t actual_batch_size);

  const ModelArgs& args_;
  const torch::Device& device_;
  const runtime::Options& options_;

  // Persistent tensors
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_new_cache_slots_;
  torch::Tensor persistent_block_tables_;
  torch::Tensor persistent_new_cache_slots_default_;
  torch::Tensor persistent_block_tables_default_;
  torch::Tensor persistent_expanded_block_tables_;
  // When q_seq_lens contains values greater than 1(chunked prefill mode or
  // speculative decode mode), the mask needs to be passed to the attention
  // operation
  torch::Tensor persistent_mask_;
  torch::Tensor persistent_mask_zero_template_;
  torch::Tensor persistent_mask_fill_template_;
  torch::Tensor hidden_states_;

  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor q_seq_lens_default_;
  torch::Tensor kv_seq_lens_default_;
  torch::Tensor expanded_kv_seq_lens_;

  // for deepseekv3.2
  torch::Tensor q_cu_seq_lens_;
  torch::Tensor q_cu_seq_lens_default_;

  // for mtp model
  torch::Tensor persistent_embedding_;
  torch::Tensor persistent_linear_state_indices_;
  torch::Tensor persistent_num_accepted_tokens_;

  // for mrope (multimodal rotary position embedding)
  bool use_mrope_ = false;

  // ModelOutput fields
  torch::Tensor aux_hidden_states_;

  // ATB context and operation for paged attention plan
  atb::Context* context_for_plan_;
  atb::Operation* custom_pa_op_for_plan_;
  aclrtStream stream_for_plan_;

  // Persistent paged attention tiling tensor on device
  torch::Tensor tiling_data_;

  // Cached attention parameters
  int32_t num_head_;
  int32_t head_dim_;

  // Flag indicating whether attention mask needs to be updated
  bool need_update_attn_mask_;
  // Flag indicating whether attention plan needs to be updated based on model
  // type
  bool need_update_attention_plan_;

  // Persistent dp/cp ep padding buffers. Pre-allocated in constructor with
  // max decode capacity so that graph capture and replay always reference
  // stable device addresses, regardless of actual vs bucket token counts.
  DpEpPaddingData persistent_dp_ep_padding_;
  CpEpPaddingData persistent_cp_ep_padding_;

  // Copy src padding data into pre-allocated persistent buffers.
  void update_persistent_dp_ep_padding(const DpEpPaddingData& src,
                                       uint32_t padded_tokens);
  void update_persistent_cp_ep_padding(const CpEpPaddingData& src,
                                       uint32_t padded_tokens);
  void replace_capture_dp_ep_padding(const DpEpPaddingData& src,
                                     DpEpPaddingData& dst) const;
  void replace_capture_cp_ep_padding(const CpEpPaddingData& src,
                                     CpEpPaddingData& dst) const;
};

}  // namespace xllm::npu

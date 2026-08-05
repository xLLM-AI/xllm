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

#pragma once

#include <acl/acl.h>
#include <torch/torch.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/kernels/npu/paged_attention_tiling_layout.h"
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

int32_t get_mla_capture_kv_seq_len_bucket(const ModelInputParams& params,
                                          const runtime::Options& options);

struct PagedAttentionPlanDescriptor {
  std::vector<uint32_t> normalized_tiling;
  uint64_t workspace_size = 0;
  kernel::npu::PagedAttentionTilingLayout layout;
};

inline bool operator==(const PagedAttentionPlanDescriptor& lhs,
                       const PagedAttentionPlanDescriptor& rhs) {
  return lhs.workspace_size == rhs.workspace_size && lhs.layout == rhs.layout &&
         lhs.normalized_tiling == rhs.normalized_tiling;
}

enum class SpecVerifyInputUpdateScope : uint8_t {
  TOKENS_ONLY,
  ALL_INPUTS,
};

// Helper class to hold persistent parameters for graph execution
// Multiple AclGraph instances can share the same GraphPersistentParam object
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       bool need_update_attn_mask = false,
                       bool is_hybrid_linear_attention = false,
                       bool supports_mla_graph_kv_bucketing = false);

  ~GraphPersistentParam();

  // Update persistent tensors with new input data
  // If return_capture_params is true, returns persistent graph inputs.
  // buffer references. During capture, pass for_capture=true so model-specific
  // host parameters can be bucketed for graph tiling/workspace. During replay,
  // return_capture_params may still be true for metadata refresh, but
  // for_capture must stay false so dynamic host metadata uses actual lengths.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& k_cache,
                                         const torch::Tensor& v_cache,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens,
                                         bool return_capture_params = false,
                                         bool skip_token_update = false,
                                         bool for_capture = false);

  void update_tokens(const torch::Tensor& tokens,
                     const ModelInputParams& params,
                     uint32_t actual_num_tokens,
                     uint32_t padded_num_tokens);

  // Update persistent graph inputs from speculative-verify source tensors on
  // the current producer stream. TileLang fusion is an optional specialization
  // hidden behind this interface.
  void update_spec_verify_inputs(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 const ModelInputParams& params,
                                 uint32_t padded_num_tokens,
                                 SpecVerifyInputUpdateScope scope);

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
  std::optional<PagedAttentionPlanDescriptor> paged_attention_plan_descriptor(
      int64_t num_rows,
      int64_t spec_width) const;
  std::optional<PagedAttentionPlanDescriptor>
  classify_spec_verify_paged_attention_plan(const torch::Tensor& tokens,
                                            const torch::Tensor& k_cache,
                                            const torch::Tensor& v_cache,
                                            const ModelInputParams& params);
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
  const int32_t* persistent_host_q_seq_lens_data() const {
    return persistent_host_q_seq_lens_.data();
  }
  const int32_t* persistent_host_kv_seq_lens_data() const {
    return persistent_host_kv_seq_lens_.data();
  }
  const int32_t* capture_host_q_seq_lens_data() const {
    return capture_host_q_seq_lens_.data();
  }
  const int32_t* capture_host_kv_seq_lens_data() const {
    return capture_host_kv_seq_lens_.data();
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
                                   aclrtStream stream,
                                   bool copy_to_device = true);

  std::vector<int32_t> update_expanded_spec_decode_attention(
      const ModelInputParams& input_params,
      uint32_t actual_num_tokens,
      uint32_t padded_num_tokens);

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
  std::vector<int32_t> persistent_host_q_seq_lens_;
  std::vector<int32_t> persistent_host_kv_seq_lens_;
  std::vector<int32_t> capture_host_q_seq_lens_;
  std::vector<int32_t> capture_host_kv_seq_lens_;

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
  std::vector<uint32_t> paged_attention_tiling_template_;
  uint64_t paged_attention_plan_workspace_size_ = 0;
  std::mutex paged_attention_plan_mutex_;

  // Cached attention parameters
  int32_t num_head_;
  int32_t head_dim_;

  // Flag indicating whether attention mask needs to be updated
  bool need_update_attn_mask_;
  // Flag indicating whether the model uses hybrid linear attention
  // (e.g., Qwen3.5/Next with gated delta net layers)
  bool is_hybrid_linear_attention_;
  // Flag indicating whether MLA graph capture uses KV length bucketing.
  bool supports_mla_graph_kv_bucketing_;
  // Flag indicating whether attention plan needs to be updated based on model
  // type
  bool need_update_attention_plan_;

  // Persistent dp/cp ep padding buffers. Pre-allocated in constructor with
  // max decode capacity so that graph capture and replay always reference
  // stable device addresses, regardless of actual vs bucket token counts.
  DpEpPaddingData persistent_dp_ep_padding_;
  CpEpMeta persistent_cp_ep_meta_;

  // Copy src padding data into pre-allocated persistent buffers.
  void update_persistent_dp_ep_padding(const DpEpPaddingData& src,
                                       uint32_t padded_tokens);
  void update_persistent_cp_ep_meta(const CpEpMeta& src,
                                    uint32_t padded_tokens);
  void replace_capture_dp_ep_padding(const DpEpPaddingData& src,
                                     DpEpPaddingData& dst) const;
  void replace_capture_cp_ep_meta(const CpEpMeta& src, CpEpMeta& dst) const;
};

}  // namespace xllm::npu

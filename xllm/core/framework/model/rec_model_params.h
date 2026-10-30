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

#include <utility>

#include "core/common/macros.h"
#include "core/framework/model/domain_attention_input.h"
#include "core/framework/model/rec_strategy_params.h"

namespace xllm {

class RecEmbeddingInput final {
 public:
  RecEmbeddingInput to(const torch::Device& device) const {
    RecEmbeddingInput out;
    out.input_embedding = safe_to(input_embedding, device);
    out.embedding_ids = embedding_ids;
    out.linear_state_ids = linear_state_ids;
    out.linear_state_indices = safe_to(linear_state_indices, device, true);
    out.linear_state_read_ids = linear_state_read_ids;
    out.linear_state_read_indices =
        safe_to(linear_state_read_indices, device, true);
    out.request_ids = request_ids;
    return out;
  }
  mutable torch::Tensor input_embedding;
  std::vector<int32_t> embedding_ids;
  std::vector<int32_t> linear_state_ids;
  torch::Tensor linear_state_indices;
  std::vector<int32_t> linear_state_read_ids;
  torch::Tensor linear_state_read_indices;
  std::vector<std::string> request_ids;
};

class RecFeatureInput final {
 public:
  RecFeatureInput to(const torch::Device& device) const {
    RecFeatureInput out;
    out.mm_data = MMBatchData::to(mm_data, device);
    return out;
  }
  mutable MMBatchData mm_data;
};

class RecGraphInput final {
 public:
  RecGraphInput to(const torch::Device& device) const {
    RecGraphInput out;
    out.attn_mask = safe_to(attn_mask, device, true);
    out.tiling_data = safe_to(tiling_data, device, true);
#if defined(USE_DCU)
    out.use_dense_flash_attention = use_dense_flash_attention;
#endif
#if defined(USE_NPU)
    out.acl_graph_task_update_context = acl_graph_task_update_context;
#endif
    return out;
  }
  torch::Tensor attn_mask;
  torch::Tensor tiling_data;
#if defined(USE_DCU)
  bool use_dense_flash_attention = false;
#endif
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext> acl_graph_task_update_context;
#endif
};

class RecModelParams final {
 public:
  RecModelParams() = default;
  RecModelParams(const RecModelParams&) = delete;
  RecModelParams& operator=(const RecModelParams&) = delete;
  RecModelParams(RecModelParams&&) = default;
  RecModelParams& operator=(RecModelParams&&) = default;

  RecModelParams clone() const {
    RecModelParams out;
    out.meta = meta;
    out.attention = attention;
    out.embedding = embedding;
    out.parallel = parallel;
    out.block_copy = block_copy;
    out.features = features;
    out.expert = expert;
    out.graph = graph;
    out.linear_state_validity_mask = linear_state_validity_mask;
    out.rec_params = rec_params;
    out.attn_metadata = attn_metadata;
    out.python_attention_metadata = python_attention_metadata;
    out.prefill_without_cache = prefill_without_cache;
    out.enable_graph = enable_graph;
    return out;
  }

  RecModelParams to(const torch::Device& device) const {
    RecModelParams params;
    params.meta = meta;
    params.prefill_without_cache = prefill_without_cache;
    params.attention = attention.to(device);
    params.embedding = embedding.to(device);
    params.parallel = parallel.to(device);
    params.block_copy = block_copy.to(device);
    params.features = features.to(device);
    params.expert = expert.to(device);
    params.graph = graph.to(device);
    params.linear_state_validity_mask = linear_state_validity_mask;
    materialize_linear_state_indices(params.embedding, device);
    if (const auto* xattention = onerec_xattention_params()) {
      params.rec_params = xattention->to(device);
    } else if (const auto* onerec = onerec_params()) {
      params.rec_params = onerec->to(device);
    } else if (const auto* llmrec = llmrec_params()) {
      params.rec_params = llmrec->to(device);
    }
#if defined(USE_MUSA)
    params.attn_metadata = attn_metadata;
#endif
    return params;
  }

  bool synchronize_layer(int64_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    CHECK_GE(layer_idx, 0) << "Layer index must be non-negative.";
    if (static_cast<uint64_t>(layer_idx) % parallel.layers_per_event == 0) {
      return parallel.layer_wise_load_synchronizer->synchronize_layer(
          layer_idx / parallel.layers_per_event);
    }
    return true;
  }

  bool record_layer(uint32_t layer_idx, const torch::Device& device) const {
#if defined(USE_MLU) || defined(USE_DCU)
    if (parallel.layer_synchronizer != nullptr) {
      return parallel.layer_synchronizer->record_current(layer_idx,
                                                         device.index());
    }
#else
    (void)layer_idx;
    (void)device;
#endif
    return true;
  }

  const OneRecModelInputParams* onerec_params() const {
    if (const auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return params;
    }
    return std::get_if<OneRecXAttentionParams>(&rec_params);
  }

  bool has_onerec_params() const { return onerec_params() != nullptr; }

  OneRecModelInputParams& mutable_onerec_params() {
    if (auto* params = std::get_if<OneRecModelInputParams>(&rec_params)) {
      return *params;
    }
    if (auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<OneRecModelInputParams>();
  }

  const OneRecXAttentionParams* onerec_xattention_params() const {
    return std::get_if<OneRecXAttentionParams>(&rec_params);
  }

  bool has_onerec_xattention_params() const {
    return onerec_xattention_params() != nullptr;
  }

  OneRecXAttentionParams& mutable_onerec_xattention_params() {
    if (auto* params = std::get_if<OneRecXAttentionParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<OneRecXAttentionParams>();
  }

  const LlmRecMultiRoundParams* llmrec_params() const {
    return std::get_if<LlmRecMultiRoundParams>(&rec_params);
  }

  bool has_llmrec_params() const { return llmrec_params() != nullptr; }

  LlmRecMultiRoundParams& mutable_llmrec_params() {
    if (auto* params = std::get_if<LlmRecMultiRoundParams>(&rec_params)) {
      return *params;
    }
    return rec_params.emplace<LlmRecMultiRoundParams>();
  }

  BatchInputMeta meta;
  RecAttentionInput attention;
  RecEmbeddingInput embedding;
  ParallelInput parallel;
  BlockCopyInput block_copy;
  RecFeatureInput features;
  ExpertInput expert;
  RecGraphInput graph;
  LinearStateValidityMask linear_state_validity_mask;
  RecModelInputParams rec_params;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  std::shared_ptr<PythonAttentionMetadata> python_attention_metadata;
  bool prefill_without_cache = false;
  bool enable_graph = false;
};

}  // namespace xllm

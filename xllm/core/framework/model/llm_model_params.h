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

#include "core/framework/model/domain_attention_input.h"
#include "core/util/tensor_helper.h"

namespace xllm {

class LlmModelParams final {
 public:
  LlmModelParams() = default;
  LlmModelParams(const LlmModelParams&) = delete;
  LlmModelParams& operator=(const LlmModelParams&) = delete;
  LlmModelParams(LlmModelParams&&) = default;
  LlmModelParams& operator=(LlmModelParams&&) = default;

  LlmModelParams clone() const {
    LlmModelParams out;
    out.meta = meta;
    out.attention = attention;
    out.embedding = embedding;
    out.parallel = parallel;
    out.block_copy = block_copy;
    out.expert = expert;
    out.graph = graph;
    out.multi_block_tables = multi_block_tables;
    out.mtp_shifted_token_ids = mtp_shifted_token_ids;
    out.linear_state_cache_ops = linear_state_cache_ops;
    out.linear_state_validity_mask = linear_state_validity_mask;
    out.is_spec_verify = is_spec_verify;
    out.prefill_without_cache = prefill_without_cache;
    out.num_accepted_tokens = num_accepted_tokens;
    out.mtp_topk_state = mtp_topk_state;
    out.num_accepted_tokens_host = num_accepted_tokens_host;
    out.attn_metadata = attn_metadata;
    out.python_attention_metadata = python_attention_metadata;
    out.enable_graph = enable_graph;
    return out;
  }

  LlmModelParams to(const torch::Device& device) const {
    LlmModelParams params;
    params.meta = meta;
    params.attention = attention.to(device);
    params.embedding = embedding.to(device);
    params.block_copy = block_copy.to(device);
    params.parallel = parallel.to(device);
    params.expert = expert.to(device);
    params.graph = graph.to(device);
    params.linear_state_cache_ops = linear_state_cache_ops;
    params.linear_state_validity_mask = linear_state_validity_mask;
    params.is_spec_verify = is_spec_verify;
    params.prefill_without_cache = prefill_without_cache;
    params.num_accepted_tokens = safe_to(num_accepted_tokens, device, true);
    params.num_accepted_tokens_host = num_accepted_tokens_host;
    params.mtp_topk_state =
        mtp_topk_state == nullptr ? nullptr : mtp_topk_state->to(device);
    params.multi_block_tables.reserve(multi_block_tables.size());
    for (const auto& table : multi_block_tables) {
      params.multi_block_tables.emplace_back(
          safe_to(table, table.options().device(torch::kCPU), true));
    }
    params.mtp_shifted_token_ids = safe_to(mtp_shifted_token_ids, device, true);
    if (!params.embedding.linear_state_indices.defined() &&
        !params.embedding.linear_state_ids.empty()) {
      params.embedding.linear_state_indices =
          make_cpu_tensor(params.embedding.linear_state_ids).to(device);
    }
#if defined(USE_MUSA)
    params.attn_metadata = attn_metadata;
#endif
    return params;
  }

  void clear_linear_attention_state() {
    embedding.linear_state_ids.clear();
    embedding.linear_state_indices = torch::Tensor();
    linear_state_cache_ops.clear();
    linear_state_validity_mask.clear();
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
    CHECK_GE(seq_idx, 0);
#if defined(USE_NPU)
    CHECK_LT(seq_idx, static_cast<int32_t>(attention.host.q_seq_lens.size()));
    return attention.host.q_seq_lens[seq_idx];
#else
    CHECK_LT(seq_idx + 1,
             static_cast<int32_t>(attention.host.q_seq_lens.size()));
    return attention.host.q_seq_lens[seq_idx + 1] -
           attention.host.q_seq_lens[seq_idx];
#endif
  }

  bool synchronize_layer(int64_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    CHECK_GE(layer_idx, 0);
    if (static_cast<uint64_t>(layer_idx) % parallel.layers_per_event == 0) {
      return parallel.layer_wise_load_synchronizer->synchronize_layer(
          layer_idx / parallel.layers_per_event);
    }
    return true;
  }

  bool synchronize_draft_layer() const {
    if (parallel.layer_wise_load_synchronizer == nullptr ||
        !parallel.draft_load_event_index.has_value()) {
      return true;
    }
    return parallel.layer_wise_load_synchronizer->synchronize_layer(
        static_cast<int64_t>(*parallel.draft_load_event_index));
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

  BatchInputMeta meta;
  LlmAttentionInput attention;
  LlmEmbeddingInput embedding;
  ParallelInput parallel;
  BlockCopyInput block_copy;
  ExpertInput expert;
  LlmGraphInput graph;
  std::vector<torch::Tensor> multi_block_tables;
  torch::Tensor mtp_shifted_token_ids;
  std::vector<LinearStateCacheOp> linear_state_cache_ops;
  LinearStateValidityMask linear_state_validity_mask;
  bool is_spec_verify = false;
  bool prefill_without_cache = false;
  torch::Tensor num_accepted_tokens;
  MtpTopkStatePtr mtp_topk_state;
  std::vector<int64_t> num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  std::shared_ptr<PythonAttentionMetadata> python_attention_metadata;
  bool enable_graph = false;
};

}  // namespace xllm

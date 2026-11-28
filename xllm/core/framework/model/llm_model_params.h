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

#include "core/framework/model/domain_attention_input.h"
#include "core/util/tensor_helper.h"

namespace xllm {

class LlmModelParams final {
 public:
  LlmModelParams() = default;
  LlmModelParams& operator=(const LlmModelParams&) = delete;
  LlmModelParams(LlmModelParams&&) = default;
  LlmModelParams& operator=(LlmModelParams&&) = default;

  LlmModelParams clone() const { return LlmModelParams(*this); }

  LlmModelParams to(const torch::Device& device) const {
    return detail::transfer_model_params(*this, device);
  }

  void clear_linear_attention_state() {
    embedding.linear_state_ids.clear();
    embedding.linear_state_indices = torch::Tensor();
    embedding.linear_state_read_ids.clear();
    embedding.linear_state_read_indices = torch::Tensor();
    linear_state_validity_mask.clear();
  }

  BatchInputMeta meta;
  LlmAttentionInput attention;
  ModelEmbeddingInput embedding;
  ParallelInput parallel;
  ExpertInput expert;
  ModelGraphInput graph;
  std::vector<torch::Tensor> multi_block_tables;
  std::vector<int64_t> linear_state_validity_mask;
  bool prefill_without_cache = false;
  torch::Tensor num_accepted_tokens;
  MtpTopkStatePtr mtp_topk_state;
  std::vector<int64_t> num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata> attn_metadata;
  bool enable_graph = false;

 private:
  LlmModelParams(const LlmModelParams&) = default;
};

}  // namespace xllm

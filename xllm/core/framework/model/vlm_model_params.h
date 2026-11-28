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

namespace xllm {

class VlmModelParams final {
 public:
  VlmModelParams() = default;
  VlmModelParams& operator=(const VlmModelParams&) = delete;
  VlmModelParams(VlmModelParams&&) = default;
  VlmModelParams& operator=(VlmModelParams&&) = default;

  VlmModelParams clone() const { return VlmModelParams(*this); }

  VlmModelParams to(const torch::Device& device) const {
    return detail::transfer_model_params(*this, device);
  }

  BatchInputMeta meta;
  VlmAttentionInput attention;
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
  VlmModelParams(const VlmModelParams&) = default;
};

}  // namespace xllm

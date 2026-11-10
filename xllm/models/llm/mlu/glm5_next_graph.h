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

#include <memory>

#include "core/common/constants.h"
#include "core/framework/model/causal_lm.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

namespace xllm::mlu::model {

// Keep captured device tensor addresses stable across graph replays.
class Glm5NextGraphMetadataState final : public ModelGraphMetadataState {
 public:
  void prepare(const torch::Tensor& positions, ModelParamsView& params);

 private:
  std::shared_ptr<layer::AttentionMetadata> metadata_;
};

inline void Glm5NextGraphMetadataState::prepare(const torch::Tensor& positions,
                                                ModelParamsView& params) {
  const auto& attention = params.attention.device;
  const size_t graph_rows = attention.kv_seq_lens.numel();
  if (!params.embedding.linear_state_ids.empty() &&
      params.embedding.linear_state_ids.size() != graph_rows) {
    params.embedding.linear_state_ids.resize(
        graph_rows, static_cast<int32_t>(kPaddingLinearStateId));
  }
  if (!metadata_) {
    metadata_ = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(params,
                                               /*enable_mla=*/true,
                                               /*compute_dtype=*/"half",
                                               /*attn_mask=*/std::nullopt,
                                               positions.device()));
    // Zero sizes the workspace from the static block-table width.
    metadata_->max_seq_len = 0;
    params.attn_metadata = metadata_;
    layer::prepare_glm5_next_kpool_metadata(*params.attn_metadata,
                                            positions.device());
    return;
  }
  params.attn_metadata = metadata_;
  auto& attn = *params.attn_metadata;
  attn.kv_seq_lens = attention.kv_seq_lens;
  attn.q_seq_lens = attention.q_seq_lens;
}

}  // namespace xllm::mlu::model

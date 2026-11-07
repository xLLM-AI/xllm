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
  torch::Tensor has_initial_states_host_;
};

inline void Glm5NextGraphMetadataState::prepare(const torch::Tensor& positions,
                                                ModelParamsView& params) {
  const auto& attention = params.attention.device;
  const size_t graph_rows = attention.kv_seq_lens.numel() - 1;
  if (!params.embedding.linear_state_ids.empty() &&
      params.embedding.linear_state_ids.size() != graph_rows) {
    params.embedding.linear_state_ids.resize(
        graph_rows, static_cast<int32_t>(kPaddingLinearStateId));
  }
  if (params.linear_state_validity_mask.size() != graph_rows) {
    params.linear_state_validity_mask.resize(graph_rows, 0);
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
    if (metadata_->has_initial_states.defined()) {
      has_initial_states_host_ = torch::empty(
          metadata_->has_initial_states.sizes(),
          metadata_->has_initial_states.options().device(torch::kCPU));
    }
    params.attn_metadata = metadata_;
    layer::prepare_glm5_next_kpool_metadata(*params.attn_metadata,
                                            positions.device());
    return;
  }
  params.attn_metadata = metadata_;
  auto& attn = *params.attn_metadata;
  torch::diff_out(attn.kv_seq_lens, attention.kv_seq_lens);
  torch::diff_out(attn.q_seq_lens, attention.q_seq_lens);
  if (attn.has_initial_states.defined()) {
    // The executor refreshes the graph-owned device mask, preferring an
    // explicit device input over the host source just like the builder.
    if (params.linear_state_validity_mask_tensor.defined()) {
      if (attn.has_initial_states.data_ptr() !=
          params.linear_state_validity_mask_tensor.data_ptr()) {
        attn.has_initial_states.copy_(params.linear_state_validity_mask_tensor);
      }
    } else {
      CHECK_EQ(has_initial_states_host_.numel(),
               static_cast<int64_t>(params.linear_state_validity_mask.size()));
      auto* staging = has_initial_states_host_.data_ptr<bool>();
      for (size_t i = 0; i < params.linear_state_validity_mask.size(); ++i) {
        staging[i] = params.linear_state_validity_mask[i] != 0;
      }
      attn.has_initial_states.copy_(has_initial_states_host_, false);
    }
  }
  // Ordinary decode aliases graph-owned storage, refreshed by the executor.
  // Expanded attention rows materialize a request-level table instead. Copy
  // its current page IDs on every replay without replacing captured storage.
  auto* batch = attn.kpool_batch_metadata.get();
  if (batch != nullptr && batch->block_table.defined() &&
      batch->block_table.data_ptr() != attention.block_tables.data_ptr()) {
    batch->block_table.copy_(attention.block_tables.index_select(
        /*dim=*/0,
        batch->query_starts.narrow(
            /*dim=*/0, /*start=*/0, batch->q_seq_lens.size())));
  }
}

}  // namespace xllm::mlu::model

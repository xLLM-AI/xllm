/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/types.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/multimodal/mm_batch_data.h"

namespace xllm {

inline torch::Tensor merge_multimodal_embeddings(torch::Tensor inputs_embeds,
                                                 const MMBatchData& mm_data) {
  auto merge_modality = [&](const std::string& embed_key,
                            const std::string& mask_key) {
    auto emb = mm_data.get<torch::Tensor>(embed_key);
    if (!emb.has_value()) {
      return;
    }
    auto mask = mm_data.get<torch::Tensor>(mask_key);
    if (!mask.has_value()) {
      return;
    }
    inputs_embeds.index_put_({mask.value()}, emb.value());
  };
  merge_modality(/*embed_key=*/"image|embedding", /*mask_key=*/"image|mask");
  merge_modality(/*embed_key=*/"video|embedding", /*mask_key=*/"video|mask");
  return inputs_embeds;
}

std::vector<int32_t> get_mm_token_nums(MMBatchData& mm_data, MMType type);

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums);

}  // namespace xllm

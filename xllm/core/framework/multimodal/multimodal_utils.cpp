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

#include "core/framework/multimodal/multimodal_utils.h"

#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {

std::vector<int32_t> get_mm_token_nums(MMBatchData& mm_data, MMType type) {
  MMTokenNumVisitor visitor(type);
  mm_data.foreach (visitor);
  return visitor.token_nums();
}

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums) {
  std::vector<int64_t> split_sizes(token_nums.begin(), token_nums.end());
  return tensor.split(split_sizes, 0);
}

}  // namespace xllm

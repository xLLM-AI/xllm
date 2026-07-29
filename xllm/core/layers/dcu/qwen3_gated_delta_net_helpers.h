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

#include <torch/torch.h>

namespace xllm {
namespace layer {
namespace qwen3_gdn {

// Small tensor helpers shared by both torch and optimized-kernel backends.

// L2-normalize along `dim` with an epsilon guard.
inline torch::Tensor l2norm(const torch::Tensor& x,
                            int64_t dim,
                            double eps = 1e-6) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

// Broadcast GQA: replicate `tensor` along `head_dim` from current head count to
// `target_heads` (target must be a multiple of current).
inline torch::Tensor repeat_tensor_heads(const torch::Tensor& tensor,
                                         int64_t target_heads,
                                         int64_t head_dim) {
  const int64_t current_heads = tensor.size(head_dim);
  if (current_heads == target_heads) {
    return tensor;
  }
  CHECK_GT(current_heads, 0) << "current heads must be positive";
  CHECK_EQ(target_heads % current_heads, 0)
      << "target heads must be divisible by current heads";

  const int64_t repeats = target_heads / current_heads;
  std::vector<int64_t> view_shape = tensor.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> output_shape = tensor.sizes().vec();
  output_shape[head_dim] = target_heads;
  return tensor.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(output_shape)
      .contiguous();
}

}  // namespace qwen3_gdn
}  // namespace layer
}  // namespace xllm

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::layer {

inline torch::Tensor build_linear_state_base_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride) {
  torch::Tensor state_indices =
      logical_state_indices.contiguous().to(torch::kInt32);
  if (checkpoint_stride == 1) {
    return state_indices;
  }
  return (state_indices * checkpoint_stride).contiguous();
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_mixed_qkv(
    const torch::Tensor& mixed_qkv,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  const int64_t num_tokens = mixed_qkv.size(0);
  const int64_t q_size = num_k_heads * head_k_dim;
  const int64_t k_size = num_k_heads * head_k_dim;
  const int64_t v_size = num_v_heads * head_v_dim;
  CHECK_EQ(mixed_qkv.size(1), q_size + k_size + v_size)
      << "MLU GDN mixed qkv channel mismatch";

  torch::Tensor q = mixed_qkv.slice(/*dim=*/1, /*start=*/0, /*end=*/q_size)
                        .contiguous()
                        .view({1, num_tokens, num_k_heads, head_k_dim});
  torch::Tensor k =
      mixed_qkv.slice(/*dim=*/1, /*start=*/q_size, /*end=*/q_size + k_size)
          .contiguous()
          .view({1, num_tokens, num_k_heads, head_k_dim});
  torch::Tensor v = mixed_qkv
                        .slice(/*dim=*/1,
                               /*start=*/q_size + k_size,
                               /*end=*/q_size + k_size + v_size)
                        .contiguous()
                        .view({1, num_tokens, num_v_heads, head_v_dim});
  return {q, k, v};
}

inline torch::Tensor build_rebased_ssm_state_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride,
    int64_t q_max_seq_len) {
  torch::Tensor base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  torch::Tensor offsets = torch::arange(q_max_seq_len, base_indices.options());
  return (base_indices.unsqueeze(/*dim=*/1) + offsets).contiguous();
}

}  // namespace xllm::layer

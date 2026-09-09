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

#include <torch/torch.h>
#include <torch/types.h>

#include <memory>

namespace xllm {
namespace layer {
class PartialRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  PartialRotaryEmbeddingImpl(int64_t rotary_dim,
                             int64_t max_position_embeddings,
                             int64_t rope_theta,
                             int64_t head_size,
                             bool is_neox_style,
                             bool interleaved,
                             const torch::TensorOptions& options);

  void forward(const torch::Tensor& positions,
               torch::Tensor& q,
               torch::Tensor& k);

  torch::Tensor get_cos_sin_cache() const { return cos_sin_cache_; }

 private:
  int64_t head_size_;
  int64_t rotary_dim_;
  bool is_neox_style_;
  bool interleaved_;
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(PartialRotaryEmbedding);

}  // namespace layer
}  // namespace xllm

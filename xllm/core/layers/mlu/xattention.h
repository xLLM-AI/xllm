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

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm::layer {

class MluXAttentionImpl : public torch::nn::Module {
 public:
  MluXAttentionImpl(int64_t num_heads,
                    int64_t head_size,
                    float scale,
                    int64_t num_kv_heads,
                    int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

 private:
  void check_common(const AttentionMetadata& attn_metadata,
                    const torch::Tensor& query,
                    const torch::Tensor& key,
                    const torch::Tensor& value,
                    const torch::Tensor& output) const;

  void prefill(const AttentionMetadata& attn_metadata,
               torch::Tensor& query,
               torch::Tensor& key,
               torch::Tensor& value,
               torch::Tensor& output) const;

  void decode(const AttentionMetadata& attn_metadata,
              torch::Tensor& query,
              torch::Tensor& key,
              torch::Tensor& value,
              torch::Tensor& output) const;

  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t num_kv_heads_;
  int64_t sliding_window_;
};

}  // namespace xllm::layer

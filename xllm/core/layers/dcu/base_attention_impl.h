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

namespace xllm {
namespace layer {

// Base class for different attention implementations.
// This class contains common member variables and defines the common interface.
class BaseAttentionImpl {
 public:
  BaseAttentionImpl(int64_t num_heads,
                    int64_t head_size,
                    float scale,
                    int64_t num_kv_heads,
                    int64_t sliding_window);

  virtual ~BaseAttentionImpl() = default;

  // Pure virtual function that must be implemented by derived classes.
  virtual std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) = 0;

 protected:
  // Common member variables shared by all attention implementations.
  int64_t num_heads_;
  int64_t head_size_;
  float scale_;
  int64_t num_kv_heads_;
  int64_t sliding_window_;
  bool decode_use_tensor_core_ = false;
};

}  // namespace layer
}  // namespace xllm

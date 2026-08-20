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

#include "attention.h"

#include "base_attention_impl.h"
#include "core/util/rec_model_utils.h"
#include "flashinfer_attention.h"
#include "xattention.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window) {
  // Select implementation based on mode. Use polymorphism via base class
  // pointer to manage different implementations.

  if (is_rec_multi_round_mode()) {
    attention_impl_ = std::make_shared<XAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  } else {
    attention_impl_ = std::make_shared<FlashInferAttentionImpl>(
        num_heads, head_size, scale, num_kv_heads, sliding_window);
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  // Create output tensor internally to unify the interface with other devices
  torch::Tensor output = torch::empty_like(query);

  // Use polymorphism to dispatch to the appropriate implementation,
  // making the code elegant and type-safe.
  return attention_impl_->forward(
      attn_metadata, query, key, value, output, kv_cache);
}

}  // namespace layer
}  // namespace xllm

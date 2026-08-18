/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/musa/attention.h"

#include <algorithm>

#include "layers/cuda/base_attention_impl.h"
#include "layers/musa/flashinfer_attention.h"

namespace xllm {
namespace layer {

BaseAttentionImpl::BaseAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads,
                                     int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window) {
  decode_use_tensor_core_ = false;
}

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window) {
  attention_impl_ = std::make_shared<FlashInferAttentionImpl>(
      num_heads, head_size, scale, num_kv_heads, sliding_window);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  torch::Tensor output;
  const bool decode_path =
      !attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill;
  if (decode_path && query.dim() >= 1 && query.numel() > 0 &&
      query.stride(-1) == 1) {
    const auto target_sizes = query.sizes();
    const int64_t last_dim = target_sizes.back();
    const int64_t target_rows = query.numel() / last_dim;
    constexpr int64_t kDefaultMaxGraphBatchSize = 256;
    const int64_t capacity_rows =
        std::max(target_rows, kDefaultMaxGraphBatchSize);
    const auto desired_options = query.options();

    const bool need_realloc =
        !output_buf_.defined() ||
        output_buf_.dtype() != desired_options.dtype() ||
        output_buf_.device() != desired_options.device() ||
        output_buf_.dim() != 2 || output_buf_.size(0) < target_rows ||
        output_buf_.size(1) != last_dim;
    if (need_realloc) {
      output_buf_ = torch::empty({capacity_rows, last_dim}, desired_options);
    }

    output = output_buf_
                 .narrow(/*dim=*/0,
                         /*start=*/0,
                         /*length=*/target_rows)
                 .view(target_sizes);
  } else {
    output = torch::empty_like(query);
  }

  return attention_impl_->forward(
      attn_metadata, query, key, value, output, kv_cache);
}

}  // namespace layer
}  // namespace xllm

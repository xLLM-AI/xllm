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

#if defined(USE_MUSA)

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
  // MUSA/Mate FlashInfer does not support the tensor-core decode path
  // (paged_run on prefill URI throws "Back optional access" for decode
  // shapes). Force the dedicated decode URI ("run" on batch_decode_*) which
  // matches the working 0526 reference path.
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
  // Provide a persistent output buffer so the FlashInfer backend can write
  // into stable storage without triggering an `at::empty_strided` allocation
  // under stream capture (forbidden on MUSA graph capture mode).
  //
  // Why this is keyed on the decode path only:
  //   * Decode is the only forward that runs under graph capture; replays
  //     reuse the captured GPU pointer to `output_buf_`, so the storage
  //     must be stable for the lifetime of every captured decode graph.
  //   * Prefill always runs eager, often at batch sizes far larger than
  //     the largest decode bucket. Routing prefill through `output_buf_`
  //     would force a re-grow whose freed storage could invalidate the
  //     captured decode pointers and corrupt later replays. Letting
  //     prefill fall back to `torch::empty_like` keeps it eager-safe and
  //     leaves the decode buffer untouched.
  //
  // Sizing strategy for the decode path: profile_manager warmups capture
  // decode buckets in descending order (largest bucket first), so the very
  // first call grows the buffer to the max bucket and every subsequent
  // smaller call simply narrows on the leading row dim. The eager
  // pre-capture warmup pass runs outside capture, which is where the
  // one-time alloc lands.
  torch::Tensor output;
  const bool decode_path =
      !attn_metadata.is_prefill && !attn_metadata.is_chunked_prefill;
  if (decode_path && query.dim() >= 1 && query.numel() > 0 &&
      query.stride(-1) == 1) {
    const auto target_sizes = query.sizes();
    const int64_t last_dim = target_sizes.back();
    const int64_t target_rows = query.numel() / last_dim;
    const auto desired_options = query.options();

    const bool need_realloc =
        !output_buf_.defined() ||
        output_buf_.dtype() != desired_options.dtype() ||
        output_buf_.device() != desired_options.device() ||
        output_buf_.dim() != query.dim() || output_buf_.size(-1) != last_dim ||
        (output_buf_.numel() / last_dim) < target_rows;
    if (need_realloc) {
      std::vector<int64_t> alloc_shape(target_sizes.begin(),
                                       target_sizes.end());
      // Grow exactly to the requested rows. Descending bucket-capture order
      // means the first warmup pass already requests the max bucket, so we
      // never realloc during stream capture.
      alloc_shape[0] = target_rows;
      output_buf_ = torch::empty(alloc_shape, desired_options);
    }

    output = (output_buf_.size(0) == target_rows)
                 ? output_buf_
                 : output_buf_.narrow(0, 0, target_rows);
  } else {
    // Prefill / chunked prefill / unusual layouts: eager allocation is
    // always legal here and avoids disturbing the captured decode buffer.
    output = torch::empty_like(query);
  }

  return attention_impl_->forward(
      attn_metadata, query, key, value, output, kv_cache);
}

}  // namespace layer
}  // namespace xllm

#endif  // USE_MUSA

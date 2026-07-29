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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/cuda/base_attention_impl.h"

namespace xllm {
namespace layer {

// MUSA-native FlashInfer attention. Mirrors the CUDA
// layers/cuda/flashinfer_attention.h declaration but is owned by the MUSA
// backend so MUSA-only state (the FA3 decode LSE scratch) stays out of the
// shared CUDA header. Only the USE_MUSA build links this class
// (layers/musa/flashinfer_attention.cpp provides the definitions); a pure
// CUDA build links its own layers/cuda copy instead, so the two declarations
// never coexist in one binary.
class FlashInferAttentionImpl final : public BaseAttentionImpl {
 public:
  FlashInferAttentionImpl(int64_t num_heads,
                          int64_t head_size,
                          float scale,
                          int64_t num_kv_heads,
                          int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) override;

 private:
  void prefill_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       std::optional<torch::Tensor>& output_lse);

  void chunked_prefill_forward(const AttentionMetadata& attn_metadata,
                               torch::Tensor& query,
                               const torch::Tensor& key,
                               torch::Tensor& output,
                               std::optional<torch::Tensor>& output_lse,
                               const torch::Tensor& k_cache,
                               const torch::Tensor& v_cache);

  void decoder_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       const torch::Tensor& key,
                       torch::Tensor& output,
                       std::optional<torch::Tensor>& output_lse,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache);

 private:
  torch::Tensor float_workspace_buffer_;
  torch::Tensor int_workspace_buffer_;
  torch::Tensor page_locked_int_workspace_buffer_;

  // Persistent grow-only scratch for the FA3 decode LSE output. The decode
  // path discards LSE (output_lse stays nullopt), but the FA3 kernel still
  // needs a writable [num_qo_heads, total_q] fp32 buffer. Serving it from a
  // persistent buffer avoids a torch::empty allocation under MUSA stream
  // capture, which is forbidden (see AttentionImpl::forward output_buf_
  // rationale).
  torch::Tensor lse_buf_;
};

}  // namespace layer
}  // namespace xllm

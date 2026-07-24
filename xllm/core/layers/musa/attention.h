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

// MUSA layer directory: FlashInfer/Mate backend for CUDA-graph path.
// Uses torch_musa runtime with mcc_wrapper + musamapping plugin.
#if defined(USE_MUSA)

#include <memory>
#include <optional>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

class BaseAttentionImpl;

// CUDA-graph attention entry for USE_MUSA (FlashInfer-only backend).
class AttentionImpl final : public torch::nn::Module {
 public:
  AttentionImpl() = default;

  AttentionImpl(int64_t num_heads,
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
  std::shared_ptr<BaseAttentionImpl> attention_impl_;

  // Caller-owned output scratch so the underlying FlashInfer backend can fill
  // its result without a per-call `at::empty_strided` call. The libtorch
  // allocation path is forbidden during MUSA stream capture; the buffer
  // lazily grows on the leading row dim (see forward() for the realloc rule),
  // then narrow()-slices for every smaller call so captured graphs hold stable
  // storage across replays.
  mutable torch::Tensor output_buf_;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace xllm

#endif  // USE_MUSA

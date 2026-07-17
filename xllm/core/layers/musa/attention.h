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

// The musa layer directory hosts two mutually exclusive attention backends
// selected at build time:
//   * XLLM_TORCH_MUSA: the CUDA-graph FlashInfer path (torch_musa runtime).
//   * USE_MUSA:        the native MTTOplib path.
// Both expose the same `AttentionImpl` / `Attention` module so downstream
// layers can include this single header regardless of the active backend.
#if defined(XLLM_TORCH_MUSA)

#include <memory>
#include <optional>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

class BaseAttentionImpl;

// CUDA-graph attention entry for XLLM_TORCH_MUSA (FlashInfer-only backend).
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

#else  // native USE_MUSA (MTTOplib backend)

#include <cassert>
#include <cstdint>
#include <optional>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/musa/layer_base.h"

namespace xllm {
namespace layer {

class AttentionImpl : public MUSALayerBaseImpl {
 public:
  explicit AttentionImpl(ModelArgs const& args,
                         QuantArgs const& quant_args,
                         ParallelArgs const& parallel_args,
                         torch::TensorOptions const& options);

  AttentionImpl(int64_t num_heads,
                int64_t head_size,
                float scale,
                int64_t num_kv_heads,
                int64_t sliding_window);

  ~AttentionImpl() {};

  torch::Tensor forward(torch::Tensor& input,
                        ForwardParams& fwd_params) override;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

  void load_state_dict(StateDict const& state_dict) override;

 private:
  int32_t num_heads_;
  int32_t num_kv_heads_;
  int32_t head_dim_;
  int32_t q_size_;
  int32_t kv_size_;
  int32_t hidden_size_;
  float rms_eps;
  float scaling_;
  constexpr static int32_t weight_num_ = 7;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace xllm

#endif  // XLLM_TORCH_MUSA

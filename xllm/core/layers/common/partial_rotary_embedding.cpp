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

#include "partial_rotary_embedding.h"

#include "kernels/ops_api.h"
#include "platform/device.h"
#include "platform/platform.h"
namespace xllm {
namespace layer {

PartialRotaryEmbeddingImpl::PartialRotaryEmbeddingImpl(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    int64_t rope_theta,
    int64_t head_size,
    bool is_neox_style,
    bool interleaved,
    const torch::TensorOptions& options)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      is_neox_style_(is_neox_style),
      interleaved_(interleaved) {
  auto dev_options = torch::TensorOptions().device(Platform::type_torch());

  auto inv_freq_t = torch::arange(/*start=*/0,
                                  /*end=*/rotary_dim_,
                                  /*step=*/2,
                                  torch::TensorOptions().dtype(torch::kFloat));
  inv_freq_t = inv_freq_t.to(dev_options);
  auto inv_freq =
      1.0 /
      torch::pow(rope_theta, inv_freq_t / static_cast<double>(rotary_dim_));

  auto t = torch::arange(0, max_position_embeddings, 1, torch::kFloat32);
  t = t.to(dev_options);

  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
  const auto cos_sin =
      torch::cat({freqs.cos(), freqs.sin()}, /*dim=*/-1).contiguous();
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
}

void PartialRotaryEmbeddingImpl::forward(const torch::Tensor& positions,
                                         torch::Tensor& q,
                                         torch::Tensor& k) {
  xllm::kernel::PartialRotaryEmbeddingParams partial_rotary_params;
  partial_rotary_params.positions = positions;
  partial_rotary_params.query = q;
  partial_rotary_params.key = k;
  partial_rotary_params.head_size = head_size_;
  partial_rotary_params.rotary_dim = rotary_dim_;
  partial_rotary_params.cos_sin_cache = cos_sin_cache_;
  partial_rotary_params.is_neox_style = is_neox_style_;
  auto [q_rot, k_rot] =
      xllm::kernel::partial_rotary_embedding(partial_rotary_params);

  q = q_rot;
  k = k_rot;
}

}  // namespace layer
}  // namespace xllm

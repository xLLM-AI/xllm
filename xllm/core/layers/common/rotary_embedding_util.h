/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <functional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"

namespace xllm {
namespace layer {
namespace rotary {

float yarn_get_mscale(float scale, float mscale);

torch::Tensor apply_deepseek_yarn_rope_scaling(float factor,
                                               int64_t extrapolation_factor,
                                               int64_t beta_fast,
                                               int64_t beta_slow,
                                               int64_t rotary_dim,
                                               float theta,
                                               int64_t old_context_len);
torch::Tensor compute_inv_freq(int64_t rotary_dim,
                               float rope_theta,
                               const torch::TensorOptions& options);
torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    float scaling_factor,
                                    float attn_factor,
                                    float mscale,
                                    float mscale_all_dim,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options);
torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options);

// Internal: Cache descriptor structure for cos_sin_cache sharing
struct CosSinCacheDesc {
  int64_t rotary_dim;
  int64_t max_position_embeddings;
  bool interleaved;
  float scaling_factor;
  float attn_factor;
  float mscale;
  float mscale_all_dim;
  size_t inv_freq_hash;  // Hash of inv_freq tensor content
  torch::Device device;
  torch::ScalarType dtype;

  bool operator==(const CosSinCacheDesc& other) const {
    return rotary_dim == other.rotary_dim &&
           max_position_embeddings == other.max_position_embeddings &&
           interleaved == other.interleaved &&
           scaling_factor == other.scaling_factor &&
           attn_factor == other.attn_factor && mscale == other.mscale &&
           mscale_all_dim == other.mscale_all_dim &&
           inv_freq_hash == other.inv_freq_hash && device == other.device &&
           dtype == other.dtype;
  }
};

// Internal: Hash function for CosSinCacheDesc
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// Internal: Compute hash for inv_freq tensor
size_t compute_inv_freq_hash(const torch::Tensor& inv_freq);

torch::Tensor get_concat_rotary_embedding(int64_t dim,
                                          int64_t seq_len,
                                          double rope_theta,
                                          const torch::TensorOptions& options);

std::pair<torch::Tensor, torch::Tensor> apply_mrope(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    const std::vector<int64_t>& mrope_section);

torch::Tensor get_chatglm_rotary_embedding(int64_t dim,
                                           int64_t seq_len,
                                           double rope_theta,
                                           const torch::TensorOptions& options);

torch::Tensor get_deepseek_rotary_embedding(
    int64_t head_size,
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    int64_t rope_scaling_original_max_position_embeddings,
    int64_t rope_theta,
    bool interleaved,
    float scaling_factor,
    float extrapolation_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    float mscale,
    float mscale_all_dim,
    const torch::TensorOptions& options);

#if defined(USE_MUSA)
torch::Tensor get_interleave_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options);
#endif
}  // namespace rotary
}  // namespace layer
}  // namespace xllm

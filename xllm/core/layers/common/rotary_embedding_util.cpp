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

#include "rotary_embedding_util.h"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <memory>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/slice.h"

namespace xllm {

namespace {

// Hash function for CosSinCacheDesc
struct CosSinCacheDescHash {
  size_t operator()(const layer::rotary::CosSinCacheDesc& key) const {
    size_t hash = 0;
    hash =
        layer::rotary::hash_combine(hash, std::hash<int64_t>{}(key.rotary_dim));
    hash = layer::rotary::hash_combine(
        hash, std::hash<int64_t>{}(key.max_position_embeddings));
    hash =
        layer::rotary::hash_combine(hash, std::hash<bool>{}(key.interleaved));
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<float>{}(key.scaling_factor));
    hash =
        layer::rotary::hash_combine(hash, std::hash<float>{}(key.attn_factor));
    hash = layer::rotary::hash_combine(hash, std::hash<float>{}(key.mscale));
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<float>{}(key.mscale_all_dim));
    hash = layer::rotary::hash_combine(hash, key.inv_freq_hash);
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<torch::Device>{}(key.device));
    hash = layer::rotary::hash_combine(
        hash, std::hash<torch::ScalarType>{}(key.dtype));
    return hash;
  }
};

// Cache manager for cos_sin_cache sharing
class CosSinCacheManager {
 public:
  static CosSinCacheManager& get_instance() {
    static CosSinCacheManager instance;
    return instance;
  }

  torch::Tensor get_or_create_cache(const layer::rotary::CosSinCacheDesc& key,
                                    std::function<torch::Tensor()> factory) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;  // Return shared tensor
    }
    // Create new and cache it
    auto tensor = factory();
    cache_[key] = tensor;
    return tensor;
  }

 private:
  std::unordered_map<layer::rotary::CosSinCacheDesc,
                     torch::Tensor,
                     CosSinCacheDescHash>
      cache_;
};

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

// Inverse dim formula to find dim based on number of rotations
inline double yarn_find_correction_dim(int num_rotations,
                                       int dim,
                                       float theta,
                                       int max_position_embeddings) {
  return (dim *
          std::log(max_position_embeddings / (num_rotations * 2 * M_PI))) /
         (2 * std::log(theta));
}

// Find dim range bounds based on rotations
inline std::tuple<int, int> yarn_find_correction_range(
    int low_rot,
    int high_rot,
    int dim,
    float theta,
    int max_position_embeddings) {
  int low = std::floor(
      yarn_find_correction_dim(low_rot, dim, theta, max_position_embeddings));
  int high = std::ceil(
      yarn_find_correction_dim(high_rot, dim, theta, max_position_embeddings));
  return std::make_tuple(std::max(low, 0), std::min(high, dim - 1));
}

// Create cos_sin tensor for rotary embedding cache
inline torch::Tensor create_cos_sin_tensor(
    int64_t max_position_embeddings,
    float scaling_factor,
    float attn_factor,
    float mscale,
    float mscale_all_dim,
    bool interleaved,
    const torch::Tensor& inv_freq,
    const torch::TensorOptions& options) {
  float mscale_ = static_cast<float>(
      layer::rotary::yarn_get_mscale(scaling_factor, mscale) /
      layer::rotary::yarn_get_mscale(scaling_factor, mscale_all_dim) *
      attn_factor);
  // [max_position_embeddings]
  auto t = torch::arange(max_position_embeddings * scaling_factor);
  // [max_position_embeddings, rotary_dim/2]
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});

  // Create cos and sin embeddings.
  torch::Tensor emd;
  if (interleaved) {
    // [a, b, c, d] => [a, a, b, b, c, c, d, d]
    emd = freqs.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  } else {
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    emd = torch::cat({freqs, freqs}, /*dim=*/-1);
  }
  const auto cos_sin =
      torch::cat({emd.cos() * mscale_, emd.sin() * mscale_}, /*dim=*/-1)
          .to(options);
  return cos_sin;
}

inline torch::Tensor yarn_linear_ramp_mask(float low, float high, int64_t dim) {
  if (low == high) {
    high += 0.001;  // Prevent singularity
  }
  torch::Tensor linear_func = torch::arange(dim) - low;
  linear_func = linear_func / (high - low);
  torch::Tensor ramp_func = torch::clamp(linear_func, 0, 1);
  return ramp_func;
}

torch::Tensor compute_rotary_embedding(int64_t dim,
                                       int64_t seq_len,
                                       double rope_theta,
                                       const torch::TensorOptions& options,
                                       bool use_cat) {
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  torch::Tensor emb;
  if (use_cat) {
    emb = torch::cat({freqs, freqs}, -1);
  } else {
    emb = torch::stack({freqs, freqs}, -1);
    emb = emb.reshape({seq_len, dim});
  }
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  std::vector<torch::Tensor> cos_sin{rope_cos, rope_sin};
  return torch::cat(cos_sin, -1);
}

#if defined(USE_MUSA)
torch::Tensor compute_rotary_embedding(int64_t dim,
                                       int64_t seq_len,
                                       double rope_theta,
                                       const torch::TensorOptions& options,
                                       bool use_cat,
                                       bool output_interleave) {
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Float);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  torch::Tensor emb;
  if (!output_interleave) {
    if (use_cat) {
      // 0, 1, ..., dim-1, 0, 1, ... dim-1
      emb = torch::cat({freqs, freqs}, -1);
    } else {
      // 0, 0, 1, 1, ..., dim-1, dim-1
      emb = torch::stack({freqs, freqs}, -1);
      emb = emb.reshape({seq_len, dim});
    }
  } else {
    // 0, 1, 2, ..., dim-1
    emb = freqs;
  }
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  std::vector<torch::Tensor> cos_sin{rope_cos, rope_sin};
  if (output_interleave) {
    // cos0, sin0, cos1, sin1 ... cos(dim-1), sin(dim-1)
    return torch::stack(cos_sin, -1).reshape({seq_len, -1});
  }
  // cos0, ... cos(dim-1), sin0, ... sin(dim-1)
  return torch::cat(cos_sin, -1);
}
#endif

}  // namespace

namespace layer {
namespace rotary {

// Compute hash for inv_freq tensor content
size_t compute_inv_freq_hash(const torch::Tensor& inv_freq) {
  // Ensure tensor is contiguous and on CPU for hashing
  auto cpu_tensor = inv_freq.contiguous().cpu();
  auto numel = cpu_tensor.numel();
  if (numel == 0) {
    return 0;
  }

  // Compute hash based on tensor data
  size_t hash = 0;
  auto data_ptr = cpu_tensor.data_ptr<float>();
  for (int64_t i = 0; i < numel; ++i) {
    // Hash each float value
    hash = hash_combine(hash, std::hash<float>{}(data_ptr[i]));
  }
  return hash;
}

float yarn_get_mscale(float scale, float mscale) {
  if (scale <= 0.0) {
    return 1.0;
  }
  return 0.1 * mscale * std::log(scale) + 1.0;
}

torch::Tensor apply_deepseek_yarn_rope_scaling(float factor,
                                               int64_t extrapolation_factor,
                                               int64_t beta_fast,
                                               int64_t beta_slow,
                                               int64_t rotary_dim,
                                               float theta,
                                               int64_t old_context_len) {
  CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
  const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
  torch::Tensor pos_freqs = torch::pow(theta, slice / rotary_dim);
  torch::Tensor inv_freq_extrapolation = 1.0 / pos_freqs;
  torch::Tensor inv_freq_interpolation = 1.0 / (factor * pos_freqs);
  int low, high;
  std::tie(low, high) = yarn_find_correction_range(
      beta_fast, beta_slow, rotary_dim, theta, old_context_len);
  // Get n-d rotational scaling corrected for extrapolation
  torch::Tensor inv_freq_mask =
      1 - yarn_linear_ramp_mask(low, high, rotary_dim / 2);
  inv_freq_mask = inv_freq_mask * extrapolation_factor;
  torch::Tensor inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) +
                           inv_freq_extrapolation * inv_freq_mask;
  return inv_freq;
}

torch::Tensor compute_inv_freq(int64_t rotary_dim,
                               float rope_theta,
                               const torch::TensorOptions& options) {
  const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
  torch::Tensor inv_freq =
      1.0 / torch::pow(rope_theta, slice / static_cast<double>(rotary_dim));
  return inv_freq;
}

torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    float scaling_factor,
                                    float attn_factor,
                                    float mscale,
                                    float mscale_all_dim,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options) {
  // Create cache descriptor using aggregate initialization
  CosSinCacheDesc desc{rotary_dim,
                       max_position_embeddings,
                       interleaved,
                       scaling_factor,
                       attn_factor,
                       mscale,
                       mscale_all_dim,
                       compute_inv_freq_hash(inv_freq),
                       options.device(),
                       options.dtype().toScalarType()};

  // Get or create cache using cache manager
  auto& cache_manager = CosSinCacheManager::get_instance();
  return cache_manager.get_or_create_cache(desc, [&]() {
    return create_cos_sin_tensor(max_position_embeddings,
                                 scaling_factor,
                                 attn_factor,
                                 mscale,
                                 mscale_all_dim,
                                 interleaved,
                                 inv_freq,
                                 options);
  });
}

torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options) {
  return compute_cos_sin_cache(rotary_dim,
                               max_position_embeddings,
                               interleaved,
                               1.0f,
                               1.0f,
                               1.0f,
                               1.0f,
                               inv_freq,
                               options);
}

torch::Tensor get_concat_rotary_embedding(int64_t dim,
                                          int64_t seq_len,
                                          double rope_theta,
                                          const torch::TensorOptions& options) {
  return compute_rotary_embedding(dim, seq_len, rope_theta, options, true);
}

namespace {

// Preserve Qwen3.5's interleaved THWTHW... layout with step-3 slice
// assignment. This avoids temporary index tensors and some contiguous copies,
// which can be faster on backends with efficient slice assignment.
std::pair<torch::Tensor, torch::Tensor> apply_sliced_mrope(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    const std::vector<int64_t>& mrope_section) {
  torch::Tensor target_cos_sin = cos_sin_cache.index({positions});
  std::vector<torch::Tensor> target_cos_sin_chunks =
      target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  auto apply = [&](const torch::Tensor& values) {
    torch::Tensor freqs = values[0].clone();
    int64_t half_rotary_dim = freqs.size(-1) / 2;

    for (int32_t axis = 1; axis <= 2; ++axis) {
      int64_t section_len = mrope_section[axis];
      if (section_len == 0) {
        continue;
      }
      int64_t slice_end = section_len * 3;
      freqs.index_put_({torch::indexing::Ellipsis,
                        torch::indexing::Slice(axis, slice_end, /*step=*/3)},
                       values[axis].index({torch::indexing::Ellipsis,
                                           torch::indexing::Slice(
                                               axis, slice_end, /*step=*/3)}));
      freqs.index_put_({torch::indexing::Ellipsis,
                        torch::indexing::Slice(axis + half_rotary_dim,
                                               slice_end + half_rotary_dim,
                                               /*step=*/3)},
                       values[axis].index(
                           {torch::indexing::Ellipsis,
                            torch::indexing::Slice(axis + half_rotary_dim,
                                                   slice_end + half_rotary_dim,
                                                   /*step=*/3)}));
    }
    return freqs;
  };
  return std::make_pair(apply(target_cos_sin_chunks[0]),
                        apply(target_cos_sin_chunks[1]));
}

std::pair<torch::Tensor, torch::Tensor> apply_indexed_mrope(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    const std::vector<int64_t>& mrope_section) {
  torch::Tensor target_cos_sin = cos_sin_cache.index({positions});
  std::vector<torch::Tensor> target_cos_sin_chunks =
      target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  torch::Tensor cos_pos = target_cos_sin_chunks[0].contiguous();
  torch::Tensor sin_pos = target_cos_sin_chunks[1].contiguous();
  torch::TensorOptions index_options = positions.options().dtype(torch::kLong);
  auto apply = [&](const torch::Tensor& values) {
    torch::Tensor freqs = values[0].clone();
    int64_t half_rotary_dim = freqs.size(-1) / 2;

    for (int32_t axis = 1; axis <= 2; ++axis) {
      int64_t section_len = mrope_section[axis];
      if (section_len == 0) {
        continue;
      }
      int64_t slice_end = section_len * 3;

      torch::Tensor idx_first_half =
          torch::arange(axis, slice_end, 3, index_options);
      torch::Tensor idx_second_half = torch::arange(axis + half_rotary_dim,
                                                    slice_end + half_rotary_dim,
                                                    3,
                                                    index_options);
      torch::Tensor idx_tensor =
          torch::cat({idx_first_half, idx_second_half}, 0).to(values.device());
      torch::Tensor src = values[axis].index_select(-1, idx_tensor);
      freqs.index_copy_(-1, idx_tensor, src);
    }
    return freqs;
  };
  cos_pos = apply(cos_pos);
  sin_pos = apply(sin_pos);
  return std::make_pair(cos_pos, sin_pos);
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> apply_mrope(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions,
    const std::vector<int64_t>& mrope_section) {
  torch::Tensor mrope_positions = positions;
  if (positions.dim() == 1) {
    mrope_positions = positions.expand({3, -1});
  }
  if (Platform::prefers_sliced_mrope()) {
    return apply_sliced_mrope(cos_sin_cache, mrope_positions, mrope_section);
  }
  return apply_indexed_mrope(cos_sin_cache, mrope_positions, mrope_section);
}

#if defined(USE_MUSA)
torch::Tensor get_interleave_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  return compute_rotary_embedding(
      dim, seq_len, rope_theta, options, false, true);
}
#endif

torch::Tensor get_chatglm_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  return compute_rotary_embedding(dim, seq_len, rope_theta, options, false);
}

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
    const torch::TensorOptions& options) {
  auto inv_freq = apply_deepseek_yarn_rope_scaling(
      scaling_factor,
      extrapolation_factor,
      beta_fast,
      beta_slow,
      rotary_dim,
      rope_theta,
      rope_scaling_original_max_position_embeddings);
  auto cos_sin = compute_cos_sin_cache(rotary_dim,
                                       max_position_embeddings,
                                       interleaved,
                                       scaling_factor,
                                       attn_factor,
                                       mscale,
                                       mscale_all_dim,
                                       inv_freq,
                                       options);
  return cos_sin;
}
}  // namespace rotary
}  // namespace layer
}  // namespace xllm

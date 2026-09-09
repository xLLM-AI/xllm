/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "rotary_embedding.h"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <memory>

#include "core/common/global_flags.h"
#include "core/util/slice.h"

namespace xllm {

namespace {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

// [1, 2, 3, 4] => [-2, 1, -4, 3]
inline torch::Tensor rotate_every_two(const torch::Tensor& x) {
  auto x1 = x.index({ISlice(), ISlice(), ISlice(0, None, 2)});
  auto x2 = x.index({ISlice(), ISlice(), ISlice(1, None, 2)});
  return torch::stack({-x2, x1}, /*dim=*/-1).flatten(/*start_dim=*/-2);
}

// apply interleaved rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor>
apply_interleaved_rotary_pos_emb(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_every_two(q) * sin);
  auto k_embed = (k * cos) + (rotate_every_two(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

// [1, 2, 3, 4] => [-3, -4, 1, 2]
inline torch::Tensor rotate_half(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

// apply rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor> apply_rotated_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_half(q) * sin);
  auto k_embed = (k * cos) + (rotate_half(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_sin,
    bool interleaved) {
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  if (interleaved) {
    return apply_interleaved_rotary_pos_emb(q, k, chunks[0], chunks[1]);
  }
  return apply_rotated_rotary_pos_emb(q, k, chunks[0], chunks[1]);
}

}  // namespace

// create right instance based on params
std::shared_ptr<NpuRotaryEmbedding> create_rotary_embedding(
    const ModelArgs& args,
    int64_t rotary_dim,
    bool interleaved,
    const torch::TensorOptions& options) {
  // rotary positional embedding
  auto inv_freq = layer::rotary::apply_deepseek_yarn_rope_scaling(
      args.rope_scaling_factor(),
      args.rope_extrapolation_factor(),
      args.rope_scaling_beta_fast(),
      args.rope_scaling_beta_slow(),
      args.rotary_dim(),
      args.rope_theta(),
      args.rope_scaling_original_max_position_embeddings());

  if (boost::iequals(args.rope_scaling_rope_type(), "deepseek_yarn")) {
    const float attn_scale = args.attn_scalar().value_or(
        static_cast<float>(args.qk_nope_head_dim() + args.qk_rope_head_dim()));
    return std::make_shared<RotaryEmbeddingDeepseekYarn>(
        args.rope_scaling_factor(),
        rotary_dim,
        args.max_position_embeddings(),
        interleaved,
        args.rope_scaling_attn_factor(),
        args.rope_scaling_mscale(),
        args.rope_scaling_mscale_all_dim(),
        inv_freq,
        options);
  } else if (boost::iequals(args.rope_scaling_rope_type(), "mrope")) {
    return std::make_shared<MRotaryEmbedding>(rotary_dim,
                                              args.max_position_embeddings(),
                                              inv_freq,
                                              interleaved,
                                              args.rope_scaling_mrope_section(),
                                              options);
  } else {
    return std::make_shared<RotaryEmbeddingGeneric>(
        rotary_dim,
        args.max_position_embeddings(),
        inv_freq,
        interleaved,
        options);
  }
}

RotaryEmbeddingGeneric::RotaryEmbeddingGeneric(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    torch::Tensor inv_freq,
    bool interleaved,
    const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim), interleaved_(interleaved) {
  const auto cos_sin = layer::rotary::compute_cos_sin_cache(
      rotary_dim, max_position_embeddings, interleaved, inv_freq, options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> RotaryEmbeddingGeneric::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  DCHECK_GE(query.size(-1), rotary_dim_);
  auto query_rotary = query.index({"...", ISlice(0, rotary_dim_)});
  auto query_pass = query.index({"...", ISlice(rotary_dim_, None)});
  auto key_rotary = key.index({"...", ISlice(0, rotary_dim_)});
  auto key_pass = key.index({"...", ISlice(rotary_dim_, None)});

  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // add a new dimension for n_heads
  cos_sin = cos_sin.unsqueeze(1);
  std::tie(query_rotary, key_rotary) =
      apply_rotary_pos_emb(query_rotary, key_rotary, cos_sin, interleaved_);
  return std::make_tuple(torch::cat({query_rotary, query_pass}, /*dim=*/-1),
                         torch::cat({key_rotary, key_pass}, /*dim=*/-1));
}

RotaryEmbeddingDeepseekYarn::RotaryEmbeddingDeepseekYarn(
    float scaling_factor,
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    bool interleaved,
    float attn_factor,
    float mscale,
    float mscale_all_dim,
    torch::Tensor inv_freq,
    const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim), interleaved_(interleaved) {
  const auto cos_sin =
      layer::rotary::compute_cos_sin_cache(rotary_dim,
                                           max_position_embeddings,
                                           interleaved,
                                           scaling_factor,
                                           attn_factor,
                                           mscale,
                                           mscale_all_dim,
                                           inv_freq,
                                           options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> RotaryEmbeddingDeepseekYarn::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  auto query_rotary = query.index({"...", ISlice(0, rotary_dim_)});
  auto query_pass = query.index({"...", ISlice(rotary_dim_, None)});
  auto key_rotary = key.index({"...", ISlice(0, rotary_dim_)});
  auto key_pass = key.index({"...", ISlice(rotary_dim_, None)});
  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // add a new dimension for n_heads
  // auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  // auto cos = chunks[0];
  // auto sin = chunks[1];
  // if (interleaved_) {
  //   cos = cos.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  //   sin = sin.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  // } else {
  //   cos = torch::cat({cos, cos}, /*dim=*/-1);
  //   sin = torch::cat({sin, sin}, /*dim=*/-1);
  // }
  // cos_sin = torch::cat({cos, sin}, /*dim=*/-1);
  cos_sin = cos_sin.unsqueeze(1);
  std::tie(query_rotary, key_rotary) =
      apply_rotary_pos_emb(query_rotary, key_rotary, cos_sin, interleaved_);
  return std::make_tuple(torch::cat({query_rotary, query_pass}, /*dim=*/-1),
                         torch::cat({key_rotary, key_pass}, /*dim=*/-1));
}

MRotaryEmbedding::MRotaryEmbedding(int64_t rotary_dim,
                                   int64_t max_position_embeddings,
                                   torch::Tensor inv_freq,
                                   bool interleaved,
                                   const std::vector<int64_t>& mrope_section,
                                   const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim),
      interleaved_(interleaved),
      mrope_section_(mrope_section) {
  // [max_position_embeddings]
  auto t = torch::arange(0, max_position_embeddings, 1, torch::kFloat32);
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

  const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> MRotaryEmbedding::forward(
    const torch::Tensor& query,  // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,    // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions) const {  // [num_tokens]

  DCHECK_GE(query.size(-1), rotary_dim_);
  auto query_rotary = query.index({"...", ISlice(0, rotary_dim_)});
  auto query_pass = query.index({"...", ISlice(rotary_dim_, None)});
  auto key_rotary = key.index({"...", ISlice(0, rotary_dim_)});
  auto key_pass = key.index({"...", ISlice(rotary_dim_, None)});

  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // mrope_section
  if (positions.dim() == 2) {
    auto chunks = cos_sin.chunk(2, -1);

    auto apply = [this](torch::Tensor x) {
      auto sections = mrope_section_;
      if (interleaved_) {
        for (auto& element : sections) {
          element *= 2;
        }
      } else {
        sections.insert(sections.end(), sections.begin(), sections.end());
      }

      auto vec = x.split(sections, -1);
      std::vector<torch::Tensor> selects;
      selects.reserve(vec.size());

      for (int64_t i = 0; i < vec.size(); ++i) {
        auto m = vec[i];
        if (interleaved_) {
          selects.push_back(m[i]);
        } else {
          selects.push_back(m[i % mrope_section_.size()]);
        }
      }
      return torch::cat(selects, -1);
    };

    auto cos = apply(chunks[0]);
    auto sin = apply(chunks[1]);
    cos_sin = torch::cat({cos, sin}, -1);
  }

  // add a new dimension for n_heads
  cos_sin = cos_sin.unsqueeze(1);
  std::tie(query_rotary, key_rotary) =
      apply_rotary_pos_emb(query_rotary, key_rotary, cos_sin, interleaved_);

  return std::make_tuple(torch::cat({query_rotary, query_pass}, /*dim=*/-1),
                         torch::cat({key_rotary, key_pass}, /*dim=*/-1));
}

}  // namespace xllm

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/types.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm {

// The frozen numerical contract, independent of either model's registration.
class MiniMaxH3TransformerConfig final {
 public:
  static constexpr int64_t kNumLayers = 50;
  static constexpr int64_t kNumRefinerLayers = 2;
  static constexpr int64_t kHiddenSize = 5376;
  static constexpr int64_t kNumAttentionHeads = 56;
  static constexpr int64_t kAttentionHeadDim = 128;
  static constexpr int64_t kFfnHiddenSize = 14336;
  static constexpr int64_t kVideoLatentDim = 24;
  static constexpr int64_t kAudioLatentDim = 32;
  static constexpr int64_t kTextDim = 5120;
  static constexpr int64_t kTimestepInputDim = 256;
  static constexpr int64_t kTimeEmbedHiddenSize = 5376;
  static constexpr int64_t kTimeEmbedDim = 2688;
  static constexpr int64_t kAdalnOutFeatures = 96768;
  static constexpr int64_t kFinalAdalnOutFeatures = 10752;
  static constexpr int64_t kRopeInvFreqLen = 16;
  static constexpr double kRopeTheta = 10000.0;
  static constexpr double kNormEps = 1e-5;
};

struct H3QualifiedShapeConfig {
  int64_t hidden_size = 5376;
  int64_t heads = 56;
  int64_t head_dim = 128;
  int64_t ffn_dim = 14336;
  int64_t layers = 50;
  int64_t refiner_layers = 2;
  int64_t video_patch_dim = 96;
  int64_t audio_dim = 32;
  int64_t text_dim = 5120;
  int64_t time_input_dim = 256;
  int64_t time_hidden_dim = 5376;
  int64_t time_dim = 2688;
  int64_t rope_freq_dim = 16;
  double rope_theta = 10000.0;
};

inline H3QualifiedShapeConfig h3_qualified_shape_from_args(
    const ModelArgs& args) {
  const H3QualifiedShapeConfig shape;
  const std::vector<int64_t> patch{1, 2, 2};
  if (args.hidden_size() != shape.hidden_size ||
      args.n_heads() != shape.heads || args.head_dim() != shape.head_dim ||
      args.ffn_dim() != shape.ffn_dim || args.num_layers() != shape.layers ||
      args.num_refiner_layers() != shape.refiner_layers ||
      args.in_channels() != 24 || args.audio_in_channels() != shape.audio_dim ||
      args.wan_patch_size() != patch || args.text_dim() != shape.text_dim ||
      args.time_freq_dim() != shape.time_input_dim ||
      args.time_embed_hidden_dim() != shape.time_hidden_dim ||
      args.time_embed_dim() != shape.time_dim ||
      args.rope_freq_dim() != shape.rope_freq_dim ||
      args.rope_theta() != shape.rope_theta ||
      !std::isfinite(args.norm_eps()) ||
      std::abs(args.norm_eps() - 1e-5) > 1e-10 ||
      !std::isfinite(args.qk_norm_eps()) ||
      std::abs(args.qk_norm_eps() - 1e-5) > 1e-10 ||
      !std::isfinite(args.final_norm_eps()) ||
      std::abs(args.final_norm_eps() - 1e-5) > 1e-10) {
    throw std::invalid_argument(
        "MiniMax-H3 qualified model geometry does not match TP2xU8 Ref2VA v1");
  }
  return shape;
}

struct H3QualifiedTensorMapping {
  std::string source;
  std::string target;
  std::vector<int64_t> shape;
  torch::ScalarType dtype;
};

inline std::vector<H3QualifiedTensorMapping> h3_qualified_fixed_mappings(
    const H3QualifiedShapeConfig& shape) {
  const int64_t h = shape.hidden_size;
  const int64_t t = shape.time_dim;
  return {
      {"proj_in.weight",
       "video_patch_proj.weight",
       {h, shape.video_patch_dim},
       torch::kFloat32},
      {"proj_in.bias", "video_patch_proj.bias", {h}, torch::kFloat32},
      {"audio_proj_in.weight",
       "audio_patch_proj.weight",
       {h, shape.audio_dim},
       torch::kFloat32},
      {"audio_proj_in.bias", "audio_patch_proj.bias", {h}, torch::kFloat32},
      {"context_embedder.weight",
       "condition_proj.weight",
       {h, shape.text_dim},
       torch::kBFloat16},
      {"context_embedder.bias", "condition_proj.bias", {h}, torch::kBFloat16},
      {"time_embedder.linear_1.weight",
       "time_embedder.proj_in.weight",
       {shape.time_hidden_dim, shape.time_input_dim},
       torch::kFloat32},
      {"time_embedder.linear_1.bias",
       "time_embedder.proj_in.bias",
       {shape.time_hidden_dim},
       torch::kFloat32},
      {"time_embedder.linear_2.weight",
       "time_embedder.proj_out.weight",
       {t, shape.time_hidden_dim},
       torch::kFloat32},
      {"time_embedder.linear_2.bias",
       "time_embedder.proj_out.bias",
       {t},
       torch::kFloat32},
      {"token_refiner.final_norm.weight",
       "token_refiner.final_norm.weight",
       {h},
       torch::kBFloat16},
      {"norm_out.norm.weight",
       "final_layer.norm.weight",
       {h},
       torch::kBFloat16},
      {"norm_out.linear.weight",
       "final_layer.adaln_proj.linear.weight",
       {2 * h, t},
       torch::kBFloat16},
      {"norm_out.linear.bias",
       "final_layer.adaln_proj.linear.bias",
       {2 * h},
       torch::kBFloat16},
      {"proj_out.weight",
       "final_layer.video_out.weight",
       {shape.video_patch_dim, h},
       torch::kFloat32},
      {"proj_out.bias",
       "final_layer.video_out.bias",
       {shape.video_patch_dim},
       torch::kFloat32},
      {"audio_proj_out.weight",
       "final_layer.audio_out.weight",
       {shape.audio_dim, h},
       torch::kFloat32},
      {"audio_proj_out.bias",
       "final_layer.audio_out.bias",
       {shape.audio_dim},
       torch::kFloat32}};
}

inline std::vector<H3QualifiedTensorMapping> h3_qualified_block_mappings(
    const H3QualifiedShapeConfig& shape,
    const std::string& source_prefix,
    const std::string& target_prefix,
    bool with_adaln) {
  const int64_t h = shape.hidden_size;
  const int64_t d = shape.head_dim;
  const int64_t inner = shape.heads * d;
  std::vector<H3QualifiedTensorMapping> mappings;
  mappings.reserve(with_adaln ? 8 : 6);
  auto add = [&](const std::string& source,
                 const std::string& target,
                 std::vector<int64_t> sizes) {
    mappings.emplace_back(H3QualifiedTensorMapping{source_prefix + source,
                                                   target_prefix + target,
                                                   std::move(sizes),
                                                   torch::kBFloat16});
  };
  add("norm1.weight", "norm1.weight", {h});
  add("norm2.weight", "norm2.weight", {h});
  add("attn.norm_q.weight", "attn.q_norm.weight", {d});
  add("attn.norm_k.weight", "attn.k_norm.weight", {d});
  add("attn.to_out.0.weight", "attn.out_proj.weight", {h, inner});
  add("ff.net.2.weight", "mlp.fc2.weight", {h, shape.ffn_dim});
  if (with_adaln) {
    add("adaln_proj.linear.weight",
        "adaln_proj.linear.weight",
        {18 * h, shape.time_dim});
    add("adaln_proj.linear.bias", "adaln_proj.linear.bias", {18 * h});
  }
  return mappings;
}

// Includes every required U key before any cross-shard conversion is attempted.
inline std::vector<H3QualifiedTensorMapping> h3_qualified_source_inventory(
    const H3QualifiedShapeConfig& shape) {
  auto inventory = h3_qualified_fixed_mappings(shape);
  inventory.reserve(18 + 10 * shape.refiner_layers + 12 * shape.layers);
  auto add_block = [&](const std::string& prefix, bool with_adaln) {
    auto mappings = h3_qualified_block_mappings(shape, prefix, "", with_adaln);
    for (auto& mapping : mappings) {
      inventory.emplace_back(std::move(mapping));
    }
    for (const std::string& projection : {"to_q", "to_k", "to_v"}) {
      inventory.emplace_back(H3QualifiedTensorMapping{
          prefix + "attn." + projection + ".weight",
          "",
          {shape.heads * shape.head_dim, shape.hidden_size},
          torch::kBFloat16});
    }
    inventory.emplace_back(
        H3QualifiedTensorMapping{prefix + "ff.net.0.proj.weight",
                                 "",
                                 {2 * shape.ffn_dim, shape.hidden_size},
                                 torch::kBFloat16});
  };
  for (int64_t index = 0; index < shape.refiner_layers; ++index) {
    add_block("token_refiner.refiner_blocks." + std::to_string(index) + ".",
              false);
  }
  for (int64_t index = 0; index < shape.layers; ++index) {
    add_block("transformer_blocks." + std::to_string(index) + ".", true);
  }
  return inventory;
}

}  // namespace xllm

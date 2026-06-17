/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {

class DeepseekV4RotaryEmbedding : public torch::nn::Module {
 public:
  using CosSinPair = std::pair<torch::Tensor, torch::Tensor>;
  using GroupCosSinMap = std::unordered_map<std::string, CosSinPair>;

  DeepseekV4RotaryEmbedding(int64_t rotary_dim,
                            int64_t max_position_embeddings,
                            bool interleaved,
                            float rope_theta,
                            float compress_rope_theta,
                            float scaling_factor,
                            float extrapolation_factor,
                            int64_t beta_fast,
                            int64_t beta_slow,
                            float attn_factor,
                            float mscale,
                            float mscale_all_dim,
                            int64_t original_max_position_embeddings,
                            const torch::TensorOptions& options);

  GroupCosSinMap build(const std::unordered_map<std::string, torch::Tensor>&
                           positions_map) const;

  GroupCosSinMap build(const torch::Tensor& default_positions) const;

  void register_layer(const std::string& layer_name,
                      const std::vector<std::string>& groups);

  GroupCosSinMap select_layer_groups(const std::string& layer_name,
                                     const GroupCosSinMap& group_cos_sin) const;

  torch::Tensor get_cos_sin_cache(const std::string& group_name) const;

  std::vector<std::string> registered_groups() const;

 private:
  torch::Tensor create_cos_sin_cache(
      float theta,
      float scaling_factor,
      float extrapolation_factor,
      int64_t beta_fast,
      int64_t beta_slow,
      float attn_factor,
      float mscale,
      float mscale_all_dim,
      int64_t original_max_position_embeddings) const;

  int64_t rotary_dim_ = 0;
  int64_t max_position_embeddings_ = 0;
  bool interleaved_ = false;
  torch::TensorOptions options_;

  std::unordered_map<std::string, torch::Tensor> cos_sin_cache_by_group_;
  std::unordered_map<std::string, std::vector<std::string>> layer_groups_;
};

}  // namespace layer
}  // namespace xllm

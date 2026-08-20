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

#pragma once

#include <map>
#include <vector>

#include "base_loader.h"

namespace xllm {
namespace layer {

class Qwen2dot5VisionEncoderLoader : public BaseLoader {
 public:
  Qwen2dot5VisionEncoderLoader(uint64_t weight_count,
                               const ModelContext& context,
                               int64_t numAttentionHeadsPerRank,
                               LoadMode mode = LoadMode::kEager);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;

 protected:
  void merge_host_at_weights() override;

 private:
  void get_weights_col_packed_qkv();
  void pad_qkv_weights();
  void pad_mlp_weights();
  torch::Tensor pad_tensor(const torch::Tensor& tensor,
                           int64_t target_shape,
                           int64_t dim = 0);

  std::vector<torch::Tensor> qkv_weight_;
  std::vector<torch::Tensor> qkv_bias_;
  int encode_param_rank_;
  int encode_param_world_size_;
  int64_t encode_param_num_attention_heads_per_rank_;
};

}  // namespace layer
}  // namespace xllm

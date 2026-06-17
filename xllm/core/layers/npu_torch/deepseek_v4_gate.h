/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <string>
#include <tuple>

#include "framework/model/model_args.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class DeepseekV4GateImpl : public torch::nn::Module {
 public:
  DeepseekV4GateImpl() = default;
  DeepseekV4GateImpl(const ModelContext& context, int32_t layer_id);
  DeepseekV4GateImpl(const ModelArgs& args,
                     int32_t layer_id,
                     const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

  void load_state_dict(const StateDict& state_dict);

  bool is_hash_layer() const { return hash_layer_; }

 private:
  std::tuple<torch::Tensor, torch::Tensor> select_experts_native(
      const torch::Tensor& router_logits,
      const std::optional<torch::Tensor>& input_ids) const;

  int64_t score_func_to_norm_type(const std::string& score_func) const;

  int64_t hidden_size_ = 0;
  int64_t topk_ = 1;
  int64_t n_routed_experts_ = 0;
  int64_t n_hash_layers_ = 0;
  double route_scale_ = 1.0;
  std::string score_func_ = "softmax";
  bool hash_layer_ = false;

  torch::Tensor weight_;
  torch::Tensor tid2eid_;
  torch::Tensor bias_;
};

TORCH_MODULE(DeepseekV4Gate);

}  // namespace layer
}  // namespace xllm

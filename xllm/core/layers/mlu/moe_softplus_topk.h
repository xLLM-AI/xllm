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

#include <torch/torch.h>

#include <optional>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class MOESoftPlusTopKImpl final : public torch::nn::Module {
 public:
  MOESoftPlusTopKImpl() = default;

  MOESoftPlusTopKImpl(int64_t n_routed_experts,
                      int64_t n_activated_experts,
                      float route_scale,
                      int64_t vocab_size,
                      bool use_hash,
                      const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& scores,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t n_routed_experts_ = 0;
  int64_t topk_ = 0;
  float route_scale_ = 1.0f;
  int64_t vocab_size_ = 0;
  bool use_hash_ = false;

  DEFINE_WEIGHT(tid2eid);
  DEFINE_WEIGHT(bias);

  // Cached optional wrappers to avoid creating new objects on each forward
  std::optional<torch::Tensor> tid2eid_opt_;
  std::optional<torch::Tensor> bias_opt_;
};
TORCH_MODULE(MOESoftPlusTopK);
}  // namespace layer
}  // namespace xllm

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

#include "layers/common/qwen3_next_rms_norm.h"

namespace xllm {
namespace layer {

class Qwen3NextRMSNormMluImpl final : public Qwen3NextRMSNormImpl {
 public:
  using Qwen3NextRMSNormImpl::Qwen3NextRMSNormImpl;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      torch::Tensor& input,
      std::optional<torch::Tensor> residual = std::nullopt) override;
};

TORCH_MODULE(Qwen3NextRMSNormMlu);

}  // namespace layer
}  // namespace xllm

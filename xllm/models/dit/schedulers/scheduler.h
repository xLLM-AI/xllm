/* Copyright 2026 The xLLM Authors.

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
#include <vector>

namespace xllm::dit {

// Abstract diffusion-scheduler interface inherited directly by concrete
// schedulers (FlowMatchEulerDiscrete, UniPCMultistep, ...).
class Scheduler : public torch::nn::Module {
 public:
  ~Scheduler() override = default;

  // Sets the inference timestep schedule; optional args may be ignored by
  // schedulers that own their schedule (e.g. the distill path).
  virtual void set_timesteps(
      int64_t num_inference_steps,
      const torch::Device& device = torch::kCPU,
      const std::optional<std::vector<float>>& sigmas = std::nullopt,
      const std::optional<float>& mu = std::nullopt,
      const std::optional<std::vector<float>>& timesteps = std::nullopt) = 0;

  virtual const torch::Tensor& timesteps() const = 0;

  virtual torch::Tensor step(const torch::Tensor& model_output,
                             const torch::Tensor& timestep,
                             const torch::Tensor& sample,
                             const std::optional<torch::Tensor>&
                                 per_token_timesteps = std::nullopt) = 0;
};

}  // namespace xllm::dit

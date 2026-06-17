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

#include "lanczos_resample.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"

namespace xllm::dit {

float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    xllm::FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

std::pair<int64_t, int64_t> calculate_dimensions(double target_area,
                                                 double ratio) {
  double width = std::sqrt(target_area * ratio);
  double height = width / ratio;

  width = std::round(width / 32) * 32;
  height = std::round(height / 32) * 32;

  return {static_cast<int64_t>(width), static_cast<int64_t>(height)};
}

torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options,
                           torch::ScalarType target_type = torch::kBFloat16) {
  if (shape.empty()) {
    LOG(FATAL) << "Shape must not be empty.";
  }
  torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>();
  gen = gen.clone();
  gen.set_current_seed(seed);
  torch::Tensor latents;
  latents =
      torch::randn(shape, gen, options.device(torch::kCPU).dtype(target_type));
  latents = latents.to(options.device());
  return latents;
}

}  // namespace xllm::dit

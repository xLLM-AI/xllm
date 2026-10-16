/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>

#include "core/framework/model_loader/dit_model_loader.h"
#include "core/util/json_reader.h"

namespace xllm {

class MiniMaxH3Scheduler final {
 public:
  explicit MiniMaxH3Scheduler(double shift = 12.0) { set_shift(shift); }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 scheduler loader must not be null";
    const std::string config_path =
        loader->model_weights_path() + "/scheduler_config.json";
    JsonReader config;
    if (!config.parse(config_path)) {
      LOG(WARNING) << "MiniMax-H3 scheduler config not found: " << config_path
                   << ", using shift=" << shift_;
      return;
    }
    if (const auto shift = config.value<double>("shift"); shift.has_value()) {
      set_shift(*shift);
    }
  }

  void set_timesteps(int64_t num_inference_steps, const torch::Device& device) {
    CHECK_GE(num_inference_steps, 1)
        << "MiniMax-H3 scheduler requires at least one inference step";
    torch::Tensor base =
        torch::linspace(1.0,
                        0.0,
                        num_inference_steps + 1,
                        torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor sigmas = shift_ * base / (1.0 + (shift_ - 1.0) * base);
    rebuild_schedule(sigmas, device);
  }

  const torch::Tensor& timesteps() const { return timesteps_; }

  torch::Tensor scale_noise(const torch::Tensor& sample,
                            const torch::Tensor& timestep,
                            const torch::Tensor& noise) const {
    torch::Tensor t = broadcast_to_sample(
        timestep.to(sample.device(), sample.scalar_type()), sample);
    return t * sample + (1.0 - t) * noise;
  }

  torch::Tensor step(const torch::Tensor& model_output,
                     const torch::Tensor& timestep,
                     const torch::Tensor& sample) {
    CHECK(timestep.is_floating_point()) << "MiniMax-H3 scheduler.step expects "
                                           "timestep value, not integer index";
    if (step_index_ < 0) {
      step_index_ = index_for_timestep(timestep);
    }
    CHECK_LT(step_index_ + 1, sigmas_.size(0))
        << "MiniMax-H3 scheduler step out of range";

    torch::Tensor sigma_from_t = broadcast_to_sample(
        1.0 - timestep.to(sample.device(), sample.scalar_type()), sample);
    torch::Tensor denoised = sample + sigma_from_t * model_output;

    torch::Dtype compute_dtype = (sample.scalar_type() == torch::kFloat16 ||
                                  sample.scalar_type() == torch::kBFloat16)
                                     ? torch::kFloat32
                                     : sample.scalar_type();
    torch::Tensor sigma =
        sigmas_[step_index_].to(sample.device(), compute_dtype);
    torch::Tensor sigma_next =
        sigmas_[step_index_ + 1].to(sample.device(), compute_dtype);
    torch::Tensor ratio = sigma_next / sigma;
    torch::Tensor prev = ratio * sample.to(compute_dtype) +
                         (1.0 - ratio) * denoised.to(compute_dtype);
    ++step_index_;
    return prev.to(sample.scalar_type());
  }

 private:
  void set_shift(double shift) {
    CHECK_GT(shift, 0.0) << "MiniMax-H3 scheduler shift must be positive";
    shift_ = shift;
  }

  int64_t index_for_timestep(const torch::Tensor& timestep) const {
    CHECK(timesteps_.defined()) << "MiniMax-H3 scheduler timesteps are not set";
    torch::Tensor matches =
        (timesteps_ == timestep.to(timesteps_.device())).nonzero();
    CHECK_GT(matches.numel(), 0) << "MiniMax-H3 timestep is not in schedule";
    return matches[0].item<int64_t>();
  }

  // Rebuilds the schedule and resets the step cursor.
  void rebuild_schedule(const torch::Tensor& sigmas,
                        const torch::Device& device) {
    sigmas_ = sigmas.to(device);
    timesteps_ =
        (1.0 - sigmas.slice(/*dim=*/0, 0, sigmas.size(0) - 1)).to(device);
    step_index_ = -1;
  }

  static torch::Tensor broadcast_to_sample(const torch::Tensor& scalar,
                                           const torch::Tensor& sample) {
    torch::Tensor broadcast = scalar;
    while (broadcast.dim() < sample.dim()) {
      broadcast = broadcast.unsqueeze(-1);
    }
    return broadcast;
  }

  double shift_{12.0};
  torch::Tensor sigmas_;
  torch::Tensor timesteps_;
  int64_t step_index_{-1};
};

}  // namespace xllm

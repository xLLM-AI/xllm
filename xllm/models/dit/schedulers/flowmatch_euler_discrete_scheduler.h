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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/model_registry.h"

namespace xllm {
class FlowMatchEulerDiscreteSchedulerImpl : public torch::nn::Module {
 public:
  explicit FlowMatchEulerDiscreteSchedulerImpl(const ModelContext& context)
      : args_(context.get_model_args()) {
    num_train_timesteps_ = args_.num_train_timesteps();
    shift_ = args_.shift();
    use_dynamic_shifting_ = args_.use_dynamic_shifting();
    base_shift_ = args_.base_shift();
    max_shift_ = args_.max_shift(),
    base_image_seq_len_ = args_.base_image_seq_len();
    max_image_seq_len_ = args_.max_image_seq_len();
    // shift_terminal_ = static_cast<int64_t>(args_.shift_terminal()) == -1 ?
    // std::nullopt : args_.shift_terminal();
    if (static_cast<int64_t>(args_.shift_terminal()) == -1) {
      shift_terminal_ = std::nullopt;
    } else {
      shift_terminal_ = args_.shift_terminal();
    }
    time_shift_type_ = "exponential";
    std::vector<float> timesteps_vec(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; ++i) {
      timesteps_vec[i] = num_train_timesteps_ - i;
    }
    torch::Tensor timesteps = torch::from_blob(
        timesteps_vec.data(), {num_train_timesteps_}, torch::kFloat32);
    torch::Tensor sigmas = timesteps / num_train_timesteps_;
    if (!use_dynamic_shifting_) {
      sigmas = shift_ * sigmas / (1 + (shift_ - 1) * sigmas);
    }
    timesteps_ = sigmas * num_train_timesteps_;
    sigmas_ = sigmas.to(torch::kCPU);
    sigma_min_ = sigmas_.index({-1}).item<float>();
    sigma_max_ = sigmas_.index({0}).item<float>();
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  void set_begin_index(int begin_index) { begin_index_ = begin_index; }
  void set_shift(float shift) { shift_ = shift; }
  int base_image_seq_len() { return base_image_seq_len_.value(); }
  int max_image_seq_len() { return max_image_seq_len_.value(); }
  float base_shift() { return base_shift_.value(); }
  float max_shift() { return max_shift_.value(); }
  int64_t order() { return order_; }

  torch::Tensor scale_noise(
      const torch::Tensor& sample,
      const torch::Tensor& timestep,
      const std::optional<torch::Tensor>& noise = std::nullopt) {
    torch::Tensor sigmas = sigmas_.to(sample.device()).to(sample.dtype());
    torch::Tensor schedule_timesteps = timesteps_.to(sample.device());
    torch::Tensor ts = timestep.to(sample.device());

    std::vector<int> step_indices;
    if (!begin_index_.has_value()) {
      for (int i = 0; i < ts.size(0); ++i) {
        step_indices.emplace_back(
            index_for_timestep(ts[i], schedule_timesteps));
      }
    } else if (step_index_.has_value()) {
      step_indices = std::vector<int>(ts.size(0), step_index_.value());
    } else {
      step_indices = std::vector<int>(ts.size(0), begin_index_.value());
    }

    torch::Tensor sigma_indices = torch::tensor(
        step_indices, torch::dtype(torch::kLong).device(sigmas.device()));
    torch::Tensor sigma = sigmas.index_select(0, sigma_indices).flatten();
    while (sigma.dim() < sample.dim()) {
      sigma.unsqueeze_(-1);
    }

    torch::Tensor noise_tensor =
        noise.has_value() ? noise.value() : torch::randn_like(sample);
    return sigma * noise_tensor + (1.0f - sigma) * sample;
  }

  torch::Tensor time_shift(float mu, float sigma, const torch::Tensor& t) {
    if (time_shift_type_ == "exponential") {
      return time_shift_exponential(mu, sigma, t);
    } else {
      return time_shift_linear(mu, sigma, t);
    }
  }

  torch::Tensor stretch_shift_to_terminal(const torch::Tensor& t) {
    if (!shift_terminal_.has_value()) {
      LOG(FATAL) << "shift_terminal is not set";
    }
    torch::Tensor one_minus_z = 1.0f - t;
    float scale_factor = one_minus_z.index({-1}).item<float>() /
                         (1.0f - shift_terminal_.value());
    return 1.0f - (one_minus_z / scale_factor);
  }

  void set_timesteps(
      int num_inference_steps,
      const torch::Device& device = torch::kCPU,
      const std::optional<std::vector<float>>& sigmas = std::nullopt,
      const std::optional<float>& mu = std::nullopt,
      const std::optional<std::vector<float>>& timesteps = std::nullopt) {
    if (use_dynamic_shifting_ && !mu.has_value()) {
      LOG(FATAL) << "mu must be provided when use_dynamic_shifting is true";
    }
    if (sigmas.has_value() && timesteps.has_value() &&
        sigmas->size() != timesteps->size()) {
      LOG(FATAL) << "sigmas and timesteps must have the same length";
    }

    int num_steps = num_inference_steps;
    if (num_steps <= 0) {
      num_steps = sigmas.has_value() ? sigmas->size() : timesteps->size();
    }

    bool is_timesteps_provided = timesteps.has_value();
    torch::Tensor ts_tensor;
    torch::Tensor sigmas_tensor;

    if (is_timesteps_provided) {
      auto* timesteps_data = const_cast<float*>(timesteps->data());
      ts_tensor = torch::from_blob(timesteps_data, {num_steps}, torch::kFloat32)
                      .clone();
    }

    if (!sigmas.has_value()) {
      if (!timesteps.has_value()) {
        std::vector<float> ts_vec(num_steps);
        float start = sigma_max_ * num_train_timesteps_;
        float end = sigma_min_ * num_train_timesteps_;
        for (int i = 0; i < num_steps; ++i) {
          ts_vec[i] = start + (end - start) * i / (num_steps - 1);
        }
        ts_tensor =
            torch::from_blob(ts_vec.data(), {num_steps}, torch::kFloat32)
                .clone();
      }
      sigmas_tensor = ts_tensor / num_train_timesteps_;
    } else {
      auto* sigmas_data = const_cast<float*>(sigmas->data());
      sigmas_tensor =
          torch::from_blob(sigmas_data, {num_steps}, torch::kFloat32).clone();
    }

    if (use_dynamic_shifting_) {
      sigmas_tensor = time_shift(mu.value(), 1.0f, sigmas_tensor);
    } else {
      sigmas_tensor =
          shift_ * sigmas_tensor / (1.0f + (shift_ - 1.0f) * sigmas_tensor);
    }

    if (shift_terminal_.has_value()) {
      sigmas_tensor = stretch_shift_to_terminal(sigmas_tensor);
    }

    if (use_karras_sigmas_) {
      sigmas_tensor = convert_to_karras(sigmas_tensor, num_steps);
    } else if (use_exponential_sigmas_) {
      sigmas_tensor = convert_to_exponential(sigmas_tensor, num_steps);
    }

    sigmas_tensor = sigmas_tensor.to(device).to(torch::kFloat32);
    if (!is_timesteps_provided) {
      ts_tensor = sigmas_tensor * num_train_timesteps_;
    } else {
      ts_tensor = ts_tensor.to(device).to(torch::kFloat32);
    }

    if (invert_sigmas_) {
      sigmas_tensor = 1.0f - sigmas_tensor;
      ts_tensor = sigmas_tensor * num_train_timesteps_;
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::ones({1}, torch::kFloat32).to(device)});
    } else {
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::zeros({1}, torch::kFloat32).to(device)});
    }

    timesteps_ = ts_tensor;
    sigmas_ = sigmas_tensor;
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  torch::Tensor step(
      const torch::Tensor& model_output,
      const torch::Tensor& timestep,
      const torch::Tensor& sample,
      float s_churn = 0.0f,
      float s_tmin = 0.0f,
      float s_tmax = std::numeric_limits<float>::infinity(),
      float s_noise = 1.0f,
      const std::optional<torch::Generator>& generator = std::nullopt,
      const std::optional<torch::Tensor>& per_token_timesteps = std::nullopt,
      bool return_dict = true) {
    if (!step_index_.has_value()) {
      init_step_index(timestep);
    }

    torch::Tensor sample_float = sample.to(torch::kFloat32);
    torch::Tensor prev_sample;

    if (per_token_timesteps.has_value()) {
      torch::Tensor per_token_sigmas =
          per_token_timesteps.value() / num_train_timesteps_;
      torch::Tensor sigmas = sigmas_.unsqueeze(1).unsqueeze(1);
      torch::Tensor lower_mask =
          sigmas < (per_token_sigmas.unsqueeze(0) - 1e-6f);
      torch::Tensor lower_sigmas = lower_mask * sigmas;
      auto max_vals = lower_sigmas.max(0);
      lower_sigmas = std::get<0>(max_vals);

      torch::Tensor current_sigma = per_token_sigmas.unsqueeze(-1);
      torch::Tensor next_sigma = lower_sigmas.unsqueeze(-1);
      torch::Tensor dt = current_sigma - next_sigma;

      if (stochastic_sampling_) {
        torch::Tensor x0 = sample_float - current_sigma * model_output;
        torch::Tensor noise;
        noise = torch::randn_like(sample_float);
        prev_sample = (1.0f - next_sigma) * x0 + next_sigma * noise;
      } else {
        prev_sample = sample_float + dt * model_output;
      }
    } else {
      int sigma_idx = step_index_.value();
      torch::Tensor sigma = sigmas_[sigma_idx];
      torch::Tensor sigma_next = sigmas_[sigma_idx + 1];
      torch::Tensor dt = sigma_next - sigma;
      if (stochastic_sampling_) {
        torch::Tensor x0 = sample_float - sigma * model_output;
        torch::Tensor noise;
        noise = torch::randn_like(sample_float);
        prev_sample = (1.0f - sigma_next) * x0 + sigma_next * noise;
      } else {
        prev_sample = sample_float + dt * model_output;
      }
    }
    step_index_ = step_index_.value() + 1;
    if (!per_token_timesteps.has_value()) {
      prev_sample = prev_sample.to(model_output.dtype());
    }
    return prev_sample;
  }

  std::optional<int> step_index() const { return step_index_; }
  std::optional<int> begin_index() const { return begin_index_; }
  const torch::Tensor& timesteps() const { return timesteps_; }
  const torch::Tensor& sigmas() const { return sigmas_; }
  int size() const { return num_train_timesteps_; }

 private:
  torch::Tensor convert_to_karras(const torch::Tensor& in_sigmas,
                                  int num_inference_steps) {
    float sigma_min = sigma_min_;
    float sigma_max = sigma_max_;
    if (in_sigmas.numel() > 0) {
      sigma_min = in_sigmas[-1].item<float>();
      sigma_max = in_sigmas[0].item<float>();
    }

    const float rho = 7.0f;
    std::vector<float> ramp(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
      ramp[i] = static_cast<float>(i) / (num_inference_steps - 1);
    }
    torch::Tensor ramp_tensor =
        torch::from_blob(ramp.data(), {num_inference_steps}, torch::kFloat32);

    float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
    float max_inv_rho = std::pow(sigma_max, 1.0f / rho);
    return torch::pow(max_inv_rho + ramp_tensor * (min_inv_rho - max_inv_rho),
                      rho);
  }

  torch::Tensor convert_to_exponential(const torch::Tensor& in_sigmas,
                                       int num_inference_steps) {
    float sigma_min = sigma_min_;
    float sigma_max = sigma_max_;
    if (in_sigmas.numel() > 0) {
      sigma_min = in_sigmas[-1].item<float>();
      sigma_max = in_sigmas[0].item<float>();
    }

    std::vector<float> exp_sigmas(num_inference_steps);
    float log_sigma_max = std::log(sigma_max);
    float log_sigma_min = std::log(sigma_min);
    for (int i = 0; i < num_inference_steps; ++i) {
      float t = static_cast<float>(i) / (num_inference_steps - 1);
      exp_sigmas[i] =
          std::exp(log_sigma_max + t * (log_sigma_min - log_sigma_max));
    }
    return torch::from_blob(
               exp_sigmas.data(), {num_inference_steps}, torch::kFloat32)
        .clone();
  }

  torch::Tensor time_shift_exponential(float mu,
                                       float sigma,
                                       const torch::Tensor& t) {
    auto exp_mu = std::exp(mu);
    return exp_mu / (exp_mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  torch::Tensor time_shift_linear(float mu,
                                  float sigma,
                                  const torch::Tensor& t) {
    return mu / (mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  void init_step_index(const torch::Tensor& timestep) {
    if (!begin_index_.has_value()) {
      torch::Tensor ts = timestep.to(timesteps_.device());
      step_index_ = index_for_timestep(ts);
    } else {
      step_index_ = begin_index_.value();
    }
  }

  int index_for_timestep(const torch::Tensor& timestep,
                         const torch::Tensor& schedule_timesteps = {}) {
    torch::Tensor sched =
        schedule_timesteps.defined() ? schedule_timesteps : timesteps_;
    torch::Tensor indices = (sched == timestep).nonzero();

    int pos = indices.size(0) > 1 ? 1 : 0;
    return indices.index({pos, 0}).item<int>();
  }

 private:
  int num_train_timesteps_;
  float shift_;
  bool use_dynamic_shifting_;
  std::optional<float> base_shift_;
  std::optional<float> max_shift_;
  std::optional<int> base_image_seq_len_;
  std::optional<int> max_image_seq_len_;
  std::optional<float> shift_terminal_;
  bool invert_sigmas_ = false;
  bool use_karras_sigmas_ = false;
  bool use_exponential_sigmas_ = false;
  bool stochastic_sampling_ = false;
  std::string time_shift_type_;

  // State variables
  torch::Tensor timesteps_;
  torch::Tensor sigmas_;
  float sigma_min_;
  float sigma_max_;
  std::optional<int> step_index_;
  std::optional<int> begin_index_;

  int64_t order_ = 1;  // default value is 1
  ModelArgs args_;
};

TORCH_MODULE(FlowMatchEulerDiscreteScheduler);

REGISTER_MODEL_ARGS(FlowMatchEulerDiscreteScheduler, [&] {
  LOAD_ARG_OR(num_train_timesteps, "num_train_timesteps", 1000);
  LOAD_ARG_OR(shift, "shift", 1);
  LOAD_ARG_OR(shift_terminal, "shift_terminal", -1);
  LOAD_ARG_OR(use_dynamic_shifting, "use_dynamic_shifting", true);
  LOAD_ARG_OR(base_shift, "base_shift", 0.5f);
  LOAD_ARG_OR(max_shift, "max_shift", 1.15f);
  LOAD_ARG_OR(base_image_seq_len, "base_image_seq_len", 256);
  LOAD_ARG_OR(max_image_seq_len, "max_image_seq_len", 4096);
});
}  // namespace xllm

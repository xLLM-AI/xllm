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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/model_registry.h"

namespace xllm {

class UniPCMultistepSchedulerImpl final : public torch::nn::Module {
 public:
  explicit UniPCMultistepSchedulerImpl(const ModelContext& context)
      : args_(context.get_model_args()) {
    num_train_timesteps_ = args_.num_train_timesteps();
    beta_start_ = args_.beta_start();
    beta_end_ = args_.beta_end();
    beta_schedule_ = args_.beta_schedule();
    trained_betas_ = args_.trained_betas();
    solver_order_ = args_.solver_order();
    prediction_type_ = args_.prediction_type();
    thresholding_ = args_.thresholding();
    dynamic_thresholding_ratio_ = args_.dynamic_thresholding_ratio();
    sample_max_value_ = args_.sample_max_value();
    predict_x0_ = args_.predict_x0();
    solver_type_ = args_.solver_type();
    if (solver_type_ != "bh1" && solver_type_ != "bh2") {
      if (solver_type_ == "midpoint" || solver_type_ == "heun" ||
          solver_type_ == "logrho") {
        solver_type_ = "bh2";
      } else {
        LOG(FATAL) << "solver_type " << solver_type_
                   << " is not implemented for UniPCMultistepScheduler";
      }
    }
    lower_order_final_ = args_.lower_order_final();
    {
      auto vec = args_.disable_corrector();
      for (auto v : vec) {
        disable_corrector_.insert(v);
      }
    }
    use_karras_sigmas_ = args_.use_karras_sigmas();
    use_exponential_sigmas_ = args_.use_exponential_sigmas();
    use_beta_sigmas_ = args_.use_beta_sigmas();
    use_flow_sigmas_ = args_.use_flow_sigmas();
    flow_shift_ = args_.flow_shift();
    timestep_spacing_ = args_.timestep_spacing();
    steps_offset_ = args_.steps_offset();
    final_sigmas_type_ = args_.final_sigmas_type();
    rescale_betas_zero_snr_ = args_.rescale_betas_zero_snr();
    use_dynamic_shifting_ = args_.use_dynamic_shifting();
    time_shift_type_ = args_.time_shift_type();

    init_betas();

    sigma_min_ = std::nullopt;
    sigma_max_ = std::nullopt;
    shift_terminal_ = std::nullopt;

    std::vector<float> timesteps_vec(num_train_timesteps_);
    for (int64_t i = 0; i < num_train_timesteps_; ++i) {
      timesteps_vec[i] = static_cast<float>(num_train_timesteps_ - 1 - i);
    }
    timesteps_ =
        torch::from_blob(
            timesteps_vec.data(), {num_train_timesteps_}, torch::kFloat32)
            .clone();

    model_outputs_.resize(solver_order_);
    timestep_list_.resize(solver_order_);

    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
    lower_order_nums_ = 0;
    this_order_ = 1;
  }

  void set_begin_index(int64_t begin_index) { begin_index_ = begin_index; }

  int64_t order() const { return order_; }

  void set_timesteps(
      int64_t num_inference_steps,
      const torch::Device& device = torch::kCPU,
      const std::optional<std::vector<float>>& sigmas = std::nullopt,
      const std::optional<float>& mu = std::nullopt) {
    if (use_dynamic_shifting_ && !mu.has_value()) {
      LOG(FATAL) << "mu must be provided when use_dynamic_shifting is true";
    }

    int64_t num_steps = num_inference_steps;
    if (sigmas.has_value()) {
      if (!use_flow_sigmas_) {
        LOG(FATAL) << "Passing `sigmas` is only supported when "
                      "`use_flow_sigmas=True`. "
                      "Please set `use_flow_sigmas=True` during scheduler "
                      "initialization.";
      }
      num_steps = sigmas->size();
    }

    torch::Tensor timesteps_tensor;
    torch::Tensor sigmas_tensor;

    if (timestep_spacing_ == "linspace") {
      torch::Tensor ts = torch::linspace(
          0, num_train_timesteps_ - 1, num_steps + 1, torch::kFloat32);
      ts = ts.round().to(torch::kInt64);
      ts = ts.flip(0).slice(0, 0, num_steps);
      timesteps_tensor = ts;
    } else if (timestep_spacing_ == "leading") {
      int64_t step_ratio = num_train_timesteps_ / (num_steps + 1);
      std::vector<int64_t> ts_vec(num_steps);
      for (int64_t i = 0; i < num_steps; ++i) {
        ts_vec[num_steps - 1 - i] =
            static_cast<int64_t>((i + 1) * step_ratio) + steps_offset_;
      }
      timesteps_tensor =
          torch::from_blob(ts_vec.data(), {num_steps}, torch::kInt64).clone();
    } else if (timestep_spacing_ == "trailing") {
      float step_ratio = static_cast<float>(num_train_timesteps_) / num_steps;
      std::vector<int64_t> ts_vec(num_steps);
      for (int64_t i = 0; i < num_steps; ++i) {
        ts_vec[i] =
            static_cast<int64_t>(num_train_timesteps_ - i * step_ratio) - 1;
      }
      timesteps_tensor =
          torch::from_blob(ts_vec.data(), {num_steps}, torch::kInt64).clone();
    } else {
      LOG(FATAL) << "timestep_spacing must be one of 'linspace', 'leading', or "
                    "'trailing'";
    }

    if (use_flow_sigmas_) {
      if (!sigmas.has_value()) {
        double start_d = 1.0 - 1.0 / static_cast<double>(num_train_timesteps_);
        double stop_d = 0.0;
        int64_t N = num_steps;
        std::vector<float> s_vec(N);
        for (int64_t i = 0; i < N; ++i) {
          double s = start_d + static_cast<double>(i) / static_cast<double>(N) *
                                   (stop_d - start_d);
          s = flow_shift_ * s / (1.0 + (flow_shift_ - 1.0) * s);
          s_vec[i] = static_cast<float>(s);
        }
        sigmas_tensor =
            torch::from_blob(s_vec.data(), {num_steps}, torch::kFloat32)
                .clone();
      } else {
        sigmas_tensor = torch::tensor(*sigmas, torch::kFloat32);
      }

      if (use_dynamic_shifting_) {
        sigmas_tensor = time_shift(mu.value(), /*sigma=*/1.0f, sigmas_tensor);
      }

      if (shift_terminal_.has_value() && shift_terminal_.value()) {
        sigmas_tensor = stretch_shift_to_terminal(sigmas_tensor);
      }

      float eps = 1e-6f;
      if (std::fabs(sigmas_tensor[0].item<float>() - 1.0f) < eps) {
        sigmas_tensor[0] = sigmas_tensor[0].item<float>() - eps;
      }

      timesteps_tensor =
          (sigmas_tensor * num_train_timesteps_).to(torch::kInt64);

      float sigma_last;
      if (final_sigmas_type_ == "sigma_min") {
        sigma_last = sigmas_tensor[-1].item<float>();
      } else if (final_sigmas_type_ == "zero") {
        sigma_last = 0.0f;
      } else {
        LOG(FATAL)
            << "final_sigmas_type must be 'zero' or 'sigma_min', but got: "
            << final_sigmas_type_;
      }
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::tensor({sigma_last}, torch::kFloat32)});
    } else {
      torch::Tensor sigmas_all =
          ((1 - alphas_cumprod_) / alphas_cumprod_).sqrt();
      auto xp = torch::arange(0, sigmas_all.size(0), torch::kFloat32);
      auto fp = sigmas_all.to(torch::kFloat32);
      auto x = timesteps_tensor.to(torch::kFloat32);
      x = torch::clamp(x, xp[0].item<float>(), xp[-1].item<float>());
      auto indices = torch::searchsorted(xp, x);
      indices = torch::clamp(indices, 1, static_cast<int64_t>(xp.size(0)) - 1);
      auto x_lo = xp.index({indices - 1});
      auto x_hi = xp.index({indices});
      auto y_lo = fp.index({indices - 1});
      auto y_hi = fp.index({indices});
      auto slope = (y_hi - y_lo) / (x_hi - x_lo);
      sigmas_tensor = y_lo + slope * (x - x_lo);

      float sigma_last;
      if (final_sigmas_type_ == "sigma_min") {
        sigma_last = std::sqrt((1 - alphas_cumprod_[0].item<float>()) /
                               alphas_cumprod_[0].item<float>());
      } else if (final_sigmas_type_ == "zero") {
        sigma_last = 0.0f;
      } else {
        LOG(FATAL)
            << "final_sigmas_type must be 'zero' or 'sigma_min', but got: "
            << final_sigmas_type_;
      }
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::tensor({sigma_last}, torch::kFloat32)});
    }

    timesteps_ = timesteps_tensor.to(device);
    sigmas_ = sigmas_tensor;
    num_inference_steps_ = num_steps;

    model_outputs_.clear();
    model_outputs_.resize(solver_order_);
    timestep_list_.clear();
    timestep_list_.resize(solver_order_);
    lower_order_nums_ = 0;
    last_sample_ = torch::Tensor();
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  torch::Tensor step(const torch::Tensor& model_output_in,
                     const torch::Tensor& timestep,
                     const torch::Tensor& sample_in) {
    auto input_dtype = sample_in.dtype();
    torch::Tensor model_output = model_output_in.to(torch::kFloat32);
    torch::Tensor sample = sample_in.to(torch::kFloat32);
    if (num_inference_steps_ <= 0) {
      LOG(FATAL)
          << "Number of inference steps is not set, run 'set_timesteps' first";
    }

    if (!step_index_.has_value()) {
      init_step_index(timestep);
    }

    bool use_corrector = step_index_.value() > 0 && last_sample_.defined() &&
                         disable_corrector_.find(step_index_.value() - 1) ==
                             disable_corrector_.end();

    torch::Tensor model_output_convert =
        convert_model_output(model_output, sample);

    if (use_corrector) {
      LOG(INFO) << "scheduler step use_corrector";
      sample = multistep_uni_c_bh_update(
          model_output_convert, last_sample_, sample, this_order_);
    }

    for (int64_t i = 0; i < solver_order_ - 1; ++i) {
      model_outputs_[i] = model_outputs_[i + 1];
      timestep_list_[i] = timestep_list_[i + 1];
    }
    model_outputs_[solver_order_ - 1] = model_output_convert;
    timestep_list_[solver_order_ - 1] = timestep;

    int64_t this_order_calc;
    if (lower_order_final_) {
      this_order_calc = std::min(
          solver_order_,
          static_cast<int64_t>(timesteps_.size(0) - step_index_.value()));
    } else {
      this_order_calc = solver_order_;
    }
    this_order_ =
        std::min(this_order_calc, static_cast<int64_t>(lower_order_nums_ + 1));

    LOG(INFO) << "self.this_order = " << this_order_;

    last_sample_ = sample;
    torch::Tensor prev_sample =
        multistep_uni_p_bh_update(model_output, sample, this_order_);

    if (lower_order_nums_ < solver_order_) {
      lower_order_nums_++;
    }

    step_index_ = step_index_.value() + 1;

    return prev_sample.to(input_dtype);
  }

  torch::Tensor scale_model_input(const torch::Tensor& sample) {
    return sample;
  }

  torch::Tensor add_noise(const torch::Tensor& original_samples,
                          const torch::Tensor& noise,
                          const torch::Tensor& timesteps) {
    torch::Tensor sigmas =
        sigmas_.to(original_samples.device()).to(original_samples.dtype());
    torch::Tensor schedule_timesteps = timesteps_.to(original_samples.device());
    torch::Tensor ts = timesteps.to(original_samples.device());

    std::vector<int64_t> step_indices;
    if (!begin_index_.has_value()) {
      for (int64_t i = 0; i < ts.size(0); ++i) {
        step_indices.emplace_back(
            index_for_timestep(ts[i], schedule_timesteps));
      }
    } else if (step_index_.has_value()) {
      step_indices = std::vector<int64_t>(ts.size(0), step_index_.value());
    } else {
      step_indices = std::vector<int64_t>(ts.size(0), begin_index_.value());
    }

    torch::Tensor sigma_indices = torch::tensor(
        step_indices, torch::dtype(torch::kLong).device(sigmas.device()));
    torch::Tensor sigma = sigmas.index_select(0, sigma_indices).flatten();
    while (sigma.dim() < original_samples.dim()) {
      sigma = sigma.unsqueeze(-1);
    }

    auto [alpha_t, sigma_t] = sigma_to_alpha_sigma_t(sigma);
    return alpha_t * original_samples + sigma_t * noise;
  }

  std::optional<int64_t> step_index() const { return step_index_; }
  std::optional<int64_t> begin_index() const { return begin_index_; }
  const torch::Tensor& timesteps() const { return timesteps_; }
  const torch::Tensor& sigmas() const { return sigmas_; }
  int64_t size() const { return num_train_timesteps_; }

 private:
  void init_betas() {
    torch::Tensor betas;
    if (!trained_betas_.empty()) {
      betas = torch::from_blob(const_cast<float*>(trained_betas_.data()),
                               {static_cast<int64_t>(trained_betas_.size())},
                               torch::kFloat32)
                  .clone();
    } else if (beta_schedule_ == "linear") {
      betas = torch::linspace(
          beta_start_, beta_end_, num_train_timesteps_, torch::kFloat32);
    } else if (beta_schedule_ == "scaled_linear") {
      betas = torch::linspace(std::sqrt(beta_start_),
                              std::sqrt(beta_end_),
                              num_train_timesteps_,
                              torch::kFloat32);
      betas = betas * betas;
    } else if (beta_schedule_ == "squaredcos_cap_v2") {
      betas = betas_for_alpha_bar(num_train_timesteps_);
    } else {
      LOG(FATAL) << "beta_schedule " << beta_schedule_ << " is not implemented";
    }

    if (rescale_betas_zero_snr_) {
      betas = rescale_zero_terminal_snr(betas);
    }

    betas_ = betas;
    alphas_ = 1.0f - betas_;
    alphas_cumprod_ = torch::cumprod(alphas_, 0);

    if (rescale_betas_zero_snr_) {
      alphas_cumprod_[alphas_cumprod_.size(0) - 1] = std::pow(2.0f, -24);
    }

    alpha_t_ = torch::sqrt(alphas_cumprod_);
    sigma_t_ = torch::sqrt(1 - alphas_cumprod_);
    lambda_t_ = torch::log(alpha_t_) - torch::log(sigma_t_);
    sigmas_all_ = torch::sqrt((1 - alphas_cumprod_) / alphas_cumprod_);
  }

  torch::Tensor betas_for_alpha_bar(int64_t num_diffusion_timesteps) {
    auto alpha_bar_fn = [](float t) -> float {
      return std::pow(std::cos((t + 0.008f) / 1.008f * M_PI / 2.0f), 2);
    };

    std::vector<float> betas_vec;
    float max_beta = 0.999f;
    for (int64_t i = 0; i < num_diffusion_timesteps; ++i) {
      float t1 = static_cast<float>(i) / num_diffusion_timesteps;
      float t2 = static_cast<float>(i + 1) / num_diffusion_timesteps;
      betas_vec.emplace_back(
          std::min(1.0f - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta));
    }
    return torch::from_blob(
               betas_vec.data(), {num_diffusion_timesteps}, torch::kFloat32)
        .clone();
  }

  torch::Tensor rescale_zero_terminal_snr(const torch::Tensor& betas) {
    torch::Tensor alphas = 1.0f - betas;
    torch::Tensor alphas_cumprod = torch::cumprod(alphas, 0);
    torch::Tensor alphas_bar_sqrt = torch::sqrt(alphas_cumprod);

    float alphas_bar_sqrt_0 = alphas_bar_sqrt[0].item<float>();
    float alphas_bar_sqrt_T =
        alphas_bar_sqrt[alphas_bar_sqrt.size(0) - 1].item<float>();

    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T;
    alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0 /
                      (alphas_bar_sqrt_0 - alphas_bar_sqrt_T);

    torch::Tensor alphas_bar = alphas_bar_sqrt * alphas_bar_sqrt;
    torch::Tensor alphas_new =
        torch::cat({alphas_bar.slice(0, 0, 1),
                    alphas_bar.slice(0, 1) / alphas_bar.slice(0, 0, -1)});
    return 1.0f - alphas_new;
  }

  std::pair<torch::Tensor, torch::Tensor> sigma_to_alpha_sigma_t(
      const torch::Tensor& sigma) {
    if (use_flow_sigmas_) {
      return {1.0f - sigma, sigma};
    } else {
      torch::Tensor alpha_t = 1.0f / torch::sqrt(sigma * sigma + 1.0f);
      return {alpha_t, sigma * alpha_t};
    }
  }

  torch::Tensor time_shift(float mu, float sigma, const torch::Tensor& t) {
    if (time_shift_type_ == "exponential") {
      return time_shift_exponential(mu, sigma, t);
    } else {
      return time_shift_linear(mu, sigma, t);
    }
  }

  torch::Tensor time_shift_exponential(float mu,
                                       float sigma,
                                       const torch::Tensor& t) {
    float exp_mu = std::exp(mu);
    return exp_mu / (exp_mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  torch::Tensor time_shift_linear(float mu,
                                  float sigma,
                                  const torch::Tensor& t) {
    return mu / (mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  torch::Tensor stretch_shift_to_terminal(const torch::Tensor& t) {
    torch::Tensor one_minus_z = 1.0f - t;
    float last_value = one_minus_z[-1].item<float>();
    float scale_factor = last_value / (1.0f - shift_terminal_.value());
    torch::Tensor stretched_t = 1.0f - (one_minus_z / scale_factor);
    return stretched_t;
  }

  int64_t index_for_timestep(const torch::Tensor& timestep,
                             const torch::Tensor& schedule_timesteps) {
    torch::Tensor indices = (schedule_timesteps == timestep).nonzero();
    if (indices.size(0) == 0) {
      return static_cast<int64_t>(timesteps_.size(0)) - 1;
    }
    int64_t pos = indices.size(0) > 1 ? 1 : 0;
    return indices[pos][0].item<int64_t>();
  }

  void init_step_index(const torch::Tensor& timestep) {
    if (!begin_index_.has_value()) {
      torch::Tensor ts = timestep.to(timesteps_.device());
      step_index_ = index_for_timestep(ts, timesteps_);
    } else {
      step_index_ = begin_index_.value();
    }
  }

  torch::Tensor convert_model_output(const torch::Tensor& model_output,
                                     const torch::Tensor& sample) {
    torch::Tensor sigma = sigmas_[step_index_.value()];
    auto [alpha_t, sigma_t] = sigma_to_alpha_sigma_t(sigma);

    if (predict_x0_) {
      torch::Tensor x0_pred;
      if (prediction_type_ == "epsilon") {
        x0_pred = (sample - sigma_t * model_output) / alpha_t;
      } else if (prediction_type_ == "sample") {
        x0_pred = model_output;
      } else if (prediction_type_ == "v_prediction") {
        x0_pred = alpha_t * sample - sigma_t * model_output;
      } else if (prediction_type_ == "flow_prediction") {
        x0_pred = sample - sigma * model_output;
      } else {
        LOG(FATAL) << "prediction_type " << prediction_type_
                   << " is not supported";
      }

      if (thresholding_) {
        x0_pred = threshold_sample(x0_pred);
      }
      return x0_pred;
    } else {
      if (prediction_type_ == "epsilon") {
        return model_output;
      } else if (prediction_type_ == "sample") {
        return (sample - alpha_t * model_output) / sigma_t;
      } else if (prediction_type_ == "v_prediction") {
        return alpha_t * model_output + sigma_t * sample;
      } else {
        LOG(FATAL) << "prediction_type " << prediction_type_
                   << " is not supported for predict_x0=false";
      }
    }
    return model_output;
  }

  torch::Tensor threshold_sample(const torch::Tensor& sample) {
    torch::Tensor sample_float = sample.to(torch::kFloat32);
    auto sizes = sample_float.sizes();
    int64_t batch_size = sizes[0];
    int64_t channels = sizes[1];

    sample_float = sample_float.reshape({batch_size, -1});
    torch::Tensor abs_sample = sample_float.abs();

    torch::Tensor s =
        torch::quantile(abs_sample, dynamic_thresholding_ratio_, 1);
    s = torch::clamp(s, 1.0f, sample_max_value_);
    s = s.unsqueeze(1);

    sample_float = torch::clamp(sample_float, -s, s) / s;
    sample_float = sample_float.reshape(sizes);
    return sample_float.to(sample.dtype());
  }

  torch::Tensor multistep_uni_p_bh_update(const torch::Tensor& model_output,
                                          const torch::Tensor& sample,
                                          int64_t order) {
    torch::Tensor m0 = model_outputs_[solver_order_ - 1];
    torch::Tensor x = sample;

    torch::Tensor sigma_t = sigmas_[step_index_.value() + 1];
    torch::Tensor sigma_s0 = sigmas_[step_index_.value()];
    auto [alpha_t, sigma_t_val] = sigma_to_alpha_sigma_t(sigma_t);
    auto [alpha_s0, sigma_s0_val] = sigma_to_alpha_sigma_t(sigma_s0);

    torch::Tensor lambda_t = torch::log(alpha_t) - torch::log(sigma_t_val);
    torch::Tensor lambda_s0 = torch::log(alpha_s0) - torch::log(sigma_s0_val);

    torch::Tensor h = lambda_t - lambda_s0;
    torch::Device device = sample.device();

    std::vector<torch::Tensor> rks;
    std::vector<torch::Tensor> D1s;
    for (int64_t i = 1; i < order; ++i) {
      int64_t si = step_index_.value() - i;
      torch::Tensor mi = model_outputs_[solver_order_ - 1 - i];
      auto [alpha_si, sigma_si] = sigma_to_alpha_sigma_t(sigmas_[si]);
      torch::Tensor lambda_si = torch::log(alpha_si) - torch::log(sigma_si);
      torch::Tensor rk = (lambda_si - lambda_s0) / h;
      rks.emplace_back(rk);
      D1s.emplace_back((mi - m0) / rk);
    }

    rks.emplace_back(torch::tensor(1.0f));
    torch::Tensor rks_tensor = torch::stack(rks).to(device);

    std::vector<torch::Tensor> R;
    std::vector<torch::Tensor> b;

    torch::Tensor hh = predict_x0_ ? -h : h;
    torch::Tensor h_phi_1 = torch::expm1(hh);
    torch::Tensor h_phi_k = h_phi_1 / hh - 1.0f;

    float factorial_i = 1.0f;
    torch::Tensor B_h;
    if (solver_type_ == "bh1") {
      B_h = hh;
    } else if (solver_type_ == "bh2") {
      B_h = torch::expm1(hh);
    } else {
      LOG(FATAL) << "solver_type " << solver_type_ << " is not implemented";
    }

    for (int64_t i = 1; i <= order; ++i) {
      R.emplace_back(torch::pow(rks_tensor, i - 1));
      b.emplace_back(h_phi_k * factorial_i / B_h);
      factorial_i *= (i + 1);
      h_phi_k = h_phi_k / hh - 1.0f / factorial_i;
    }

    torch::Tensor R_tensor = torch::stack(R);
    torch::Tensor b_tensor = torch::stack(b).to(device);

    torch::Tensor rhos_p;
    if (D1s.size() > 0) {
      torch::Tensor D1s_tensor = torch::stack(D1s, 1);
      if (order == 2) {
        rhos_p = torch::tensor({0.5f}, sample.dtype()).to(device);
      } else {
        torch::Tensor R_sub = R_tensor.slice(0, 0, R_tensor.size(0) - 1)
                                  .slice(1, 0, R_tensor.size(1) - 1);
        torch::Tensor b_sub = b_tensor.slice(0, 0, b_tensor.size(0) - 1);
        rhos_p = torch::linalg_solve(R_sub, b_sub).to(sample.dtype());
      }
    }

    torch::Tensor x_t_;
    if (predict_x0_) {
      x_t_ = sigma_t_val / sigma_s0_val * x - alpha_t * h_phi_1 * m0;
    } else {
      x_t_ = alpha_t / alpha_s0 * x - sigma_t_val * h_phi_1 * m0;
    }

    torch::Tensor x_t;
    if (D1s.size() > 0) {
      torch::Tensor pred_res =
          torch::einsum("k,bkc...->bc...", {rhos_p, torch::stack(D1s, 1)});
      if (predict_x0_) {
        x_t = x_t_ - alpha_t * B_h * pred_res;
      } else {
        x_t = x_t_ - sigma_t_val * B_h * pred_res;
      }
    } else {
      x_t = x_t_;
    }

    return x_t.to(x.dtype());
  }

  torch::Tensor multistep_uni_c_bh_update(
      const torch::Tensor& this_model_output,
      const torch::Tensor& last_sample,
      const torch::Tensor& this_sample,
      int64_t order) {
    torch::Tensor m0 = model_outputs_[solver_order_ - 1];
    torch::Tensor x = last_sample;
    torch::Tensor x_t = this_sample;
    torch::Tensor model_t = this_model_output;

    torch::Tensor sigma_t = sigmas_[step_index_.value()];
    torch::Tensor sigma_s0 = sigmas_[step_index_.value() - 1];
    auto [alpha_t, sigma_t_val] = sigma_to_alpha_sigma_t(sigma_t);
    auto [alpha_s0, sigma_s0_val] = sigma_to_alpha_sigma_t(sigma_s0);

    torch::Tensor lambda_t = torch::log(alpha_t) - torch::log(sigma_t_val);
    torch::Tensor lambda_s0 = torch::log(alpha_s0) - torch::log(sigma_s0_val);

    torch::Tensor h = lambda_t - lambda_s0;
    torch::Device device = this_sample.device();

    std::vector<torch::Tensor> rks;
    std::vector<torch::Tensor> D1s;
    for (int64_t i = 1; i < order; ++i) {
      int64_t si = step_index_.value() - (i + 1);
      torch::Tensor mi = model_outputs_[solver_order_ - 1 - i];
      auto [alpha_si, sigma_si] = sigma_to_alpha_sigma_t(sigmas_[si]);
      torch::Tensor lambda_si = torch::log(alpha_si) - torch::log(sigma_si);
      torch::Tensor rk = (lambda_si - lambda_s0) / h;
      rks.emplace_back(rk);
      D1s.emplace_back((mi - m0) / rk);
    }

    rks.emplace_back(torch::tensor(1.0f));
    torch::Tensor rks_tensor = torch::stack(rks).to(device);

    std::vector<torch::Tensor> R;
    std::vector<torch::Tensor> b;

    torch::Tensor hh = predict_x0_ ? -h : h;
    torch::Tensor h_phi_1 = torch::expm1(hh);
    torch::Tensor h_phi_k = h_phi_1 / hh - 1.0f;

    float factorial_i = 1.0f;
    torch::Tensor B_h;
    if (solver_type_ == "bh1") {
      B_h = hh;
    } else if (solver_type_ == "bh2") {
      B_h = torch::expm1(hh);
    } else {
      LOG(FATAL) << "solver_type " << solver_type_ << " is not implemented";
    }

    for (int64_t i = 1; i <= order; ++i) {
      R.emplace_back(torch::pow(rks_tensor, i - 1));
      b.emplace_back(h_phi_k * factorial_i / B_h);
      factorial_i *= (i + 1);
      h_phi_k = h_phi_k / hh - 1.0f / factorial_i;
    }

    torch::Tensor R_tensor = torch::stack(R);
    torch::Tensor b_tensor = torch::stack(b).to(device);

    torch::Tensor rhos_c;
    if (order == 1) {
      rhos_c = torch::tensor({0.5f}, x.dtype()).to(device);
    } else {
      rhos_c = torch::linalg_solve(R_tensor, b_tensor).to(x.dtype());
    }

    torch::Tensor x_t_;
    if (predict_x0_) {
      x_t_ = sigma_t_val / sigma_s0_val * x - alpha_t * h_phi_1 * m0;
    } else {
      x_t_ = alpha_t / alpha_s0 * x - sigma_t_val * h_phi_1 * m0;
    }

    torch::Tensor x_t_result;
    if (D1s.size() > 0) {
      torch::Tensor D1s_tensor = torch::stack(D1s, 1);
      torch::Tensor corr_res =
          torch::einsum("k,bkc...->bc...",
                        {rhos_c.slice(0, 0, rhos_c.size(0) - 1), D1s_tensor});
      torch::Tensor D1_t = model_t - m0;
      if (predict_x0_) {
        x_t_result = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t);
      } else {
        x_t_result = x_t_ - sigma_t_val * B_h * (corr_res + rhos_c[-1] * D1_t);
      }
    } else {
      torch::Tensor D1_t = model_t - m0;
      if (predict_x0_) {
        x_t_result = x_t_ - alpha_t * B_h * rhos_c[-1] * D1_t;
      } else {
        x_t_result = x_t_ - sigma_t_val * B_h * rhos_c[-1] * D1_t;
      }
    }

    return x_t_result.to(x.dtype());
  }

 private:
  int64_t num_train_timesteps_;
  float beta_start_;
  float beta_end_;
  std::string beta_schedule_;
  std::vector<float> trained_betas_;
  int64_t solver_order_;
  std::string prediction_type_;
  bool thresholding_;
  float dynamic_thresholding_ratio_;
  float sample_max_value_;
  bool predict_x0_;
  std::string solver_type_;
  bool lower_order_final_;
  bool use_karras_sigmas_;
  bool use_exponential_sigmas_;
  bool use_beta_sigmas_;
  bool use_flow_sigmas_;
  float flow_shift_;
  std::string timestep_spacing_;
  int64_t steps_offset_;
  std::string final_sigmas_type_;
  bool rescale_betas_zero_snr_;
  bool use_dynamic_shifting_;
  std::string time_shift_type_;
  std::optional<float> sigma_min_;
  std::optional<float> sigma_max_;
  std::optional<bool> shift_terminal_;

  torch::Tensor betas_;
  torch::Tensor alphas_;
  torch::Tensor alphas_cumprod_;
  torch::Tensor alpha_t_;
  torch::Tensor sigma_t_;
  torch::Tensor lambda_t_;
  torch::Tensor sigmas_all_;

  torch::Tensor timesteps_;
  torch::Tensor sigmas_;
  int64_t num_inference_steps_ = 0;

  std::vector<torch::Tensor> model_outputs_;
  std::vector<torch::Tensor> timestep_list_;
  torch::Tensor last_sample_;
  int64_t lower_order_nums_;
  int64_t this_order_;
  std::unordered_set<int64_t> disable_corrector_;

  std::optional<int64_t> step_index_;
  std::optional<int64_t> begin_index_;

  int64_t order_ = 1;
  ModelArgs args_;
};

TORCH_MODULE(UniPCMultistepScheduler);

REGISTER_MODEL_ARGS(UniPCMultistepScheduler, [&] {
  LOAD_ARG_OR(num_train_timesteps, "num_train_timesteps", 1000);
  LOAD_ARG_OR(beta_start, "beta_start", 0.0001f);
  LOAD_ARG_OR(beta_end, "beta_end", 0.02f);
  LOAD_ARG_OR(beta_schedule, "beta_schedule", "linear");
  LOAD_ARG_OR(trained_betas, "trained_betas", std::vector<float>{});
  LOAD_ARG_OR(solver_order, "solver_order", 2);
  LOAD_ARG_OR(prediction_type, "prediction_type", "flow_prediction");
  LOAD_ARG_OR(thresholding, "thresholding", false);
  LOAD_ARG_OR(dynamic_thresholding_ratio, "dynamic_thresholding_ratio", 0.995f);
  LOAD_ARG_OR(sample_max_value, "sample_max_value", 1.0f);
  LOAD_ARG_OR(predict_x0, "predict_x0", true);
  LOAD_ARG_OR(solver_type, "solver_type", "bh2");
  LOAD_ARG_OR(lower_order_final, "lower_order_final", true);
  LOAD_ARG_OR(disable_corrector, "disable_corrector", std::vector<int64_t>{});
  LOAD_ARG_OR(use_karras_sigmas, "use_karras_sigmas", false);
  LOAD_ARG_OR(use_exponential_sigmas, "use_exponential_sigmas", false);
  LOAD_ARG_OR(use_beta_sigmas, "use_beta_sigmas", false);
  LOAD_ARG_OR(use_flow_sigmas, "use_flow_sigmas", true);
  LOAD_ARG_OR(flow_shift, "flow_shift", 5.0f);
  LOAD_ARG_OR(timestep_spacing, "timestep_spacing", "linspace");
  LOAD_ARG_OR(steps_offset, "steps_offset", 0);
  LOAD_ARG_OR(final_sigmas_type, "final_sigmas_type", "zero");
  LOAD_ARG_OR(rescale_betas_zero_snr, "rescale_betas_zero_snr", false);
  LOAD_ARG_OR(use_dynamic_shifting, "use_dynamic_shifting", false);
  LOAD_ARG_OR(time_shift_type, "time_shift_type", "exponential");
  LOAD_ARG_OR(sigma_min, "sigma_min", 0.0f);
  LOAD_ARG_OR(sigma_max, "sigma_max", 0.0f);
  LOAD_ARG_OR(shift_terminal, "shift_terminal", false);
});

}  // namespace xllm

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
#include <vector>

#include "models/dit/utils/util.h"

namespace xllm::dit {

class DiagonalGaussianDistribution final {
 public:
  explicit DiagonalGaussianDistribution(torch::Tensor parameters,
                                        bool deterministic = false)
      : parameters_(std::move(parameters)), deterministic_(deterministic) {
    auto chunks = parameters_.chunk(2, 1);
    mean_ = chunks[0];
    logvar_ = chunks[1];

    logvar_ = torch::clamp(logvar_, -30.0f, 20.0f);

    std_ = torch::exp(0.5f * logvar_);
    var_ = torch::exp(logvar_);

    if (deterministic_) {
      std_.fill_(0.0f);
      var_.fill_(0.0f);
    }
  }

  torch::Tensor sample(int64_t seed) const {
    torch::TensorOptions options = mean_.options();
    std::vector<int64_t> shape(mean_.sizes().begin(), mean_.sizes().end());
    return mean_ + std_ * randn_tensor(shape, seed, options);
  }

  torch::Tensor kl(const std::optional<DiagonalGaussianDistribution>& other =
                       std::nullopt) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }

    if (!other.has_value()) {
      return 0.5f * torch::sum(torch::pow(mean_, 2) + var_ - 1.0f - logvar_,
                               {1, 2, 3});
    } else {
      const auto& other_dist = other.value();
      return 0.5f * torch::sum(torch::pow(mean_ - other_dist.mean_, 2) /
                                       other_dist.var_ +
                                   var_ / other_dist.var_ - 1.0f - logvar_ +
                                   other_dist.logvar_,
                               {1, 2, 3});
    }
  }

  torch::Tensor nll(const torch::Tensor& sample,
                    const std::vector<int64_t>& dims = {1, 2, 3}) const {
    if (deterministic_) {
      return torch::tensor(0.0f, mean_.options());
    }
    const float logtwopi = std::log(2.0f * M_PI);
    return 0.5f *
           torch::sum(logtwopi + logvar_ + torch::pow(sample - mean_, 2) / var_,
                      dims);
  }

  torch::Tensor mode() const { return mean_; }

  const torch::Tensor& mean() const { return mean_; }
  const torch::Tensor& std() const { return std_; }
  const torch::Tensor& var() const { return var_; }
  const torch::Tensor& logvar() const { return logvar_; }

 private:
  torch::Tensor parameters_;
  torch::Tensor mean_;
  torch::Tensor logvar_;
  torch::Tensor std_;
  torch::Tensor var_;
  bool deterministic_;
};

}  // namespace xllm::dit
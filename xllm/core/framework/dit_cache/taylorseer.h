/* Copyright 2025-2026 The xLLM Authors.

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

#include <cmath>
#include <vector>

#include "dit_cache_impl.h"

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class TaylorSeer : public DitCacheImpl {
 public:
  TaylorSeer() = default;
  ~TaylorSeer() = default;

  TaylorSeer(const TaylorSeer&) = delete;
  TaylorSeer& operator=(const TaylorSeer&) = delete;
  TaylorSeer(TaylorSeer&&) = default;
  TaylorSeer& operator=(TaylorSeer&&) = default;

  void init(const DiTCacheConfig& cfg,
            const ParallelArgs& parallel_args) override;

  // Reset all cached derivatives and internal state
  void reset_cache();

  // Mark the beginning of a new inference step
  void mark_step_begin();

  // Compute the approximate value for the current step
  torch::Tensor approximate_value();

  // Update internal caches with the new observation Y
  void update(const torch::Tensor& Y);

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

 private:
  // Compute approximate derivatives of Y using previous steps
  std::pair<std::vector<torch::Tensor>, std::vector<bool>>
  approximate_derivative(const torch::Tensor& Y);

 private:
  bool use_cache_ = false;
  int n_derivatives_;
  int order_;
  int skip_interval_steps_;
  int last_non_approximated_step_;

  std::vector<torch::Tensor> dY_prev_;
  std::vector<torch::Tensor> dY_current_;
  std::vector<bool> valid_prev_;
  std::vector<bool> valid_current_;
};

}  // namespace xllm

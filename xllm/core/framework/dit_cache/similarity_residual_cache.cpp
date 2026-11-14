/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/dit_cache/similarity_residual_cache.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xllm {

SimilarityResidualCacheState::SimilarityResidualCacheState(
    const CacheDiTOptions& options)
    : options_(options) {
  if (options_.warmup_steps < 0 ||
      !std::isfinite(options_.residual_diff_threshold) ||
      options_.residual_diff_threshold <= 0.0F ||
      options_.max_cached_steps < -1 || options_.max_consecutive_hits <= 0 ||
      options_.front_blocks <= 0 || options_.back_blocks < 0) {
    throw std::invalid_argument(
        "similarity residual cache options are invalid");
  }
}

SimilarityResidualCacheDecision SimilarityResidualCacheState::decide(
    int64_t step,
    const torch::Tensor& front_residual,
    int64_t used_rows,
    ProcessGroup* consensus_group,
    const torch::Tensor& row_indices) {
  if (step < 0 || !front_residual.defined() || front_residual.dim() < 1 ||
      used_rows <= 0 || used_rows > front_residual.size(0)) {
    throw std::invalid_argument(
        "similarity residual cache front residual is invalid");
  }

  SimilarityResidualCacheDecision decision;
  const bool below_total_limit =
      options_.max_cached_steps < 0 || cache_hits_ < options_.max_cached_steps;
  const bool below_consecutive_limit =
      consecutive_hits_ < options_.max_consecutive_hits;
  if (step >= options_.warmup_steps && previous_front_residual_.defined() &&
      below_total_limit && below_consecutive_limit) {
    if (previous_front_residual_.sizes() != front_residual.sizes() ||
        previous_front_residual_.device() != front_residual.device() ||
        previous_front_residual_.scalar_type() !=
            front_residual.scalar_type()) {
      throw std::invalid_argument(
          "similarity residual cache tensor metadata changed");
    }
    torch::Tensor current = front_residual.narrow(0, 0, used_rows);
    torch::Tensor previous = previous_front_residual_.narrow(0, 0, used_rows);
    torch::Tensor previous_previous;
    if (previous_previous_front_residual_.defined()) {
      if (previous_previous_front_residual_.sizes() != front_residual.sizes() ||
          previous_previous_front_residual_.device() !=
              front_residual.device() ||
          previous_previous_front_residual_.scalar_type() !=
              front_residual.scalar_type()) {
        throw std::invalid_argument(
            "similarity residual cache tensor metadata changed");
      }
      previous_previous =
          previous_previous_front_residual_.narrow(0, 0, used_rows);
    }
    if (row_indices.defined()) {
      if (row_indices.dim() != 1 ||
          row_indices.scalar_type() != torch::kInt64 ||
          row_indices.device() != front_residual.device() ||
          (row_indices.numel() > 0 &&
           (row_indices.min().item<int64_t>() < 0 ||
            row_indices.max().item<int64_t>() >= used_rows))) {
        throw std::invalid_argument(
            "similarity residual cache row indices are invalid");
      }
      current = current.index_select(0, row_indices);
      previous = previous.index_select(0, row_indices);
      if (previous_previous.defined()) {
        previous_previous = previous_previous.index_select(0, row_indices);
      }
    }
    torch::Tensor prediction_numerator = torch::zeros(
        {},
        torch::TensorOptions().dtype(torch::kFloat32).device(current.device()));
    torch::Tensor prediction_denominator =
        torch::zeros_like(prediction_numerator);
    if (previous_previous.defined()) {
      const torch::Tensor current_delta =
          (current - previous).to(torch::kFloat32);
      const torch::Tensor historical_delta =
          (previous - previous_previous).to(torch::kFloat32);
      prediction_numerator = (current_delta * historical_delta).sum();
      prediction_denominator = historical_delta.square().sum();
    }
    torch::Tensor statistics =
        torch::stack({(current - previous).abs().to(torch::kFloat32).sum(),
                      previous.abs().to(torch::kFloat32).sum(),
                      torch::full({},
                                  static_cast<float>(current.numel()),
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(current.device())),
                      prediction_numerator,
                      prediction_denominator});
    if (consensus_group != nullptr && consensus_group->world_size() > 1) {
      consensus_group->allreduce(statistics);
    }
    if (statistics[2].item<float>() <= 0.0F) {
      throw std::invalid_argument(
          "similarity residual cache selected no tensor values");
    }
    decision.relative_l1 =
        (statistics[0] / (statistics[1] + 1e-6F)).item<float>();
    if (statistics[4].item<float>() > 1e-6F) {
      decision.prediction_scale =
          std::clamp((statistics[3] / statistics[4]).item<float>(), 0.0F, 2.0F);
    }
    ++similarity_checks_;
    decision.hit = std::isfinite(decision.relative_l1) &&
                   decision.relative_l1 < options_.residual_diff_threshold &&
                   stack_residual_.defined();
  }

  if (decision.hit) {
    ++cache_hits_;
    ++consecutive_hits_;
    prediction_scale_ = decision.prediction_scale;
  } else {
    previous_previous_front_residual_ = previous_front_residual_;
    previous_front_residual_ = front_residual.detach().clone();
    consecutive_hits_ = 0;
    prediction_scale_ = 0.0F;
  }
  return decision;
}

void SimilarityResidualCacheState::record_dense(
    int64_t step,
    const torch::Tensor& front_output,
    const torch::Tensor& full_output) {
  if (step < 0 || (stack_step_ >= 0 && step <= stack_step_) ||
      !front_output.defined() || !full_output.defined() ||
      front_output.sizes() != full_output.sizes() ||
      front_output.device() != full_output.device() ||
      front_output.scalar_type() != full_output.scalar_type()) {
    throw std::invalid_argument(
        "similarity residual cache dense outputs do not match");
  }
  previous_stack_residual_ = stack_residual_;
  previous_stack_step_ = stack_step_;
  stack_residual_ = (full_output - front_output).detach();
  stack_step_ = step;
  ++dense_forwards_;
}

torch::Tensor SimilarityResidualCacheState::apply(
    int64_t step,
    const torch::Tensor& front_output) const {
  if (!front_output.defined() || !stack_residual_.defined() ||
      step <= stack_step_ || front_output.sizes() != stack_residual_.sizes() ||
      front_output.device() != stack_residual_.device() ||
      front_output.scalar_type() != stack_residual_.scalar_type()) {
    throw std::logic_error(
        "similarity residual cache has no compatible stack residual");
  }
  torch::Tensor residual = stack_residual_;
  if (options_.linear_residual_prediction &&
      previous_stack_residual_.defined() && previous_stack_step_ >= 0 &&
      stack_step_ > previous_stack_step_) {
    residual = stack_residual_ +
               prediction_scale_ * (stack_residual_ - previous_stack_residual_);
  }
  return front_output + residual;
}

}  // namespace xllm

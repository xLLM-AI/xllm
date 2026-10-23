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

#include "core/scheduler/profile/dynamic_chunk_predictor.h"

#include <glog/logging.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace xllm {
namespace {
constexpr int32_t kMinFitSamples = 8;
constexpr int32_t kNumCoefficients = 3;
}  // namespace

DynamicChunkPredictor::DynamicChunkPredictor(int32_t base_chunk,
                                             int32_t min_chunk,
                                             double smooth_factor,
                                             int32_t alignment,
                                             int32_t pipeline_depth)
    : base_chunk_(base_chunk),
      min_chunk_(min_chunk),
      smooth_factor_(smooth_factor),
      alignment_(alignment),
      pipeline_depth_(pipeline_depth) {
  CHECK_GT(pipeline_depth_, 0);
  CHECK_GT(alignment_, 0);
  CHECK_GE(base_chunk_, alignment_);
  CHECK_GT(min_chunk_, 0);
  CHECK_LE(min_chunk_, base_chunk_);
  CHECK(std::isfinite(smooth_factor_));
  CHECK_GT(smooth_factor_, 0.0);
  CHECK_LE(smooth_factor_, 1.0);
}

bool DynamicChunkPredictor::fit(
    const std::vector<std::pair<int32_t, double>>& samples) {
  ready_ = false;
  if (samples.size() < kMinFitSamples) {
    return false;
  }
  const int32_t count = static_cast<int32_t>(samples.size());
  Eigen::MatrixXd design(count, kNumCoefficients);
  Eigen::VectorXd times(count);
  // Normalize lengths to keep the quadratic fit well-conditioned for long
  // contexts. Coefficients below are expressed in units of base_chunk.
  for (int32_t i = 0; i < count; ++i) {
    const auto& [length, latency] = samples[i];
    if (length <= 0 || !std::isfinite(latency) || latency <= 0.0) {
      return false;
    }
    const double length_ratio = static_cast<double>(length) / base_chunk_;
    design(i, 0) = length_ratio * length_ratio;
    design(i, 1) = length_ratio;
    design(i, 2) = 1.0;
    times(i) = latency;
  }
  if (design.colPivHouseholderQr().rank() != kNumCoefficients) {
    return false;
  }

  // Exact nonnegative least squares for three coefficients: enumerate the
  // active sets. This also admits a linear model when attention curvature
  // cannot be distinguished from measurement noise.
  Eigen::Vector3d best = Eigen::Vector3d::Zero();
  double best_error = std::numeric_limits<double>::infinity();
  for (int32_t mask = 1; mask < (1 << kNumCoefficients); ++mask) {
    std::vector<int32_t> columns;
    columns.reserve(kNumCoefficients);
    for (int32_t column = 0; column < kNumCoefficients; ++column) {
      if ((mask & (1 << column)) != 0) {
        columns.emplace_back(column);
      }
    }
    Eigen::MatrixXd active(count, columns.size());
    for (size_t i = 0; i < columns.size(); ++i) {
      active.col(i) = design.col(columns[i]);
    }
    Eigen::VectorXd fit = active.colPivHouseholderQr().solve(times);
    if (!fit.allFinite() || (fit.array() < 0.0).any()) {
      continue;
    }
    const double error = (active * fit - times).squaredNorm();
    if (error >= best_error) {
      continue;
    }
    best_error = error;
    best.setZero();
    for (size_t i = 0; i < columns.size(); ++i) {
      best(columns[i]) = fit(i);
    }
  }
  target_latency_ms_ = best(0) + best(1);
  if (!std::isfinite(target_latency_ms_) ||
      target_latency_ms_ <=
          std::numeric_limits<double>::epsilon() * times.maxCoeff() * count) {
    return false;
  }
  quadratic_ = best(0) / base_chunk_ / base_chunk_;
  linear_ = best(1) / base_chunk_;
  ready_ = true;
  return true;
}

size_t DynamicChunkPredictor::predict(size_t history,
                                      size_t remaining,
                                      size_t budget) const {
  const size_t cap =
      std::min({remaining, budget, static_cast<size_t>(base_chunk_)});
  if (!ready_) {
    return cap == remaining ? cap : cap / alignment_ * alignment_;
  }
  const double slope =
      2.0 * quadratic_ * static_cast<double>(history) + linear_;
  // Rationalized positive root avoids subtracting nearly equal values when
  // history is large. It also works for the linear case (quadratic_ == 0).
  const double root =
      2.0 * target_latency_ms_ /
      (slope +
       std::sqrt(slope * slope + 4.0 * quadratic_ * target_latency_ms_));
  const double smoothed = base_chunk_ + smooth_factor_ * (root - base_chunk_);
  const size_t desired =
      static_cast<size_t>(std::clamp(smoothed,
                                     static_cast<double>(min_chunk_),
                                     static_cast<double>(base_chunk_)));
  const size_t bounded =
      std::min(cap, std::max(desired, static_cast<size_t>(alignment_)));
  // A final partial block must be allowed to finish, including very short
  // prompts. Other chunks obey KV/CP alignment, even under a tight budget.
  if (bounded == remaining) {
    return bounded;
  }
  return bounded / alignment_ * alignment_;
}

std::vector<uint32_t> DynamicChunkPredictor::plan(size_t history,
                                                  size_t remaining,
                                                  size_t budget) const {
  std::vector<uint32_t> chunks;
  chunks.reserve(pipeline_depth_);
  for (int32_t slot = 0; slot < pipeline_depth_ && remaining > 0; ++slot) {
    const size_t tokens = predict(history, remaining, budget);
    if (tokens == 0) {
      break;
    }
    chunks.emplace_back(static_cast<uint32_t>(tokens));
    history += tokens;
    remaining -= tokens;
    budget -= tokens;
  }
  return chunks;
}

size_t DynamicChunkPredictor::predict_budget(size_t history,
                                             size_t remaining,
                                             size_t budget) const {
  const auto chunks = plan(history, remaining, budget);
  return std::accumulate(chunks.begin(), chunks.end(), size_t{0});
}

}  // namespace xllm

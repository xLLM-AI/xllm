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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {

// Startup-fitted f(L) = a*L^2 + b*L + c. Selects x such that
// f(history + x) - f(history) = f(base_chunk) - f(0).
// The predictor owns no device state and does not modify scheduler budgets.
class DynamicChunkPredictor final {
 public:
  DynamicChunkPredictor(int32_t base_chunk,
                        int32_t min_chunk,
                        double smooth_factor,
                        int32_t alignment,
                        int32_t pipeline_depth = 1);

  // Invalid or uninformative measurements leave the predictor unready.
  bool fit(const std::vector<std::pair<int32_t, double>>& samples);
  bool ready() const { return ready_; }
  double target_latency_ms() const { return target_latency_ms_; }
  size_t predict(size_t history, size_t remaining, size_t budget) const;
  std::vector<uint32_t> plan(size_t history,
                             size_t remaining,
                             size_t budget) const;
  size_t predict_budget(size_t history, size_t remaining, size_t budget) const;
  int32_t pipeline_depth() const { return pipeline_depth_; }

 private:
  int32_t base_chunk_;
  int32_t min_chunk_;
  double smooth_factor_;
  int32_t alignment_;
  int32_t pipeline_depth_;
  double quadratic_ = 0.0;
  double linear_ = 0.0;
  double target_latency_ms_ = 0.0;
  bool ready_ = false;
};

}  // namespace xllm

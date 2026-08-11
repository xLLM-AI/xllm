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

#include <cstdint>
#include <vector>

namespace xllm::worker_service_detail {

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  int64_t committed_tokens = 0;
  bool supported_dtype = false;
};

template <typename T>
SpeculativeOutputStats calculate_speculative_output_stats_typed(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  const T* data = tokens.const_data_ptr<T>();
  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  SpeculativeOutputStats stats;
  stats.accepted_per_position.assign(
      static_cast<size_t>(num_speculative_tokens), 0);
  stats.supported_dtype = true;
  for (int64_t row = 0; row < batch_size; ++row) {
    for (int64_t column = 0; column < token_width; ++column) {
      if (data[row * token_width + column] >= static_cast<T>(0)) {
        ++stats.committed_tokens;
        // Column 0 is always the first committed token. Draft position p was
        // accepted exactly when output column p+1 is non-negative.
        if (column > 0 && column <= num_speculative_tokens) {
          ++stats.accepted_per_position[static_cast<size_t>(column - 1)];
        }
      }
    }
  }
  return stats;
}

inline SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  switch (tokens.scalar_type()) {
    case torch::kInt64:
      return calculate_speculative_output_stats_typed<int64_t>(
          tokens, num_speculative_tokens);
    case torch::kInt32:
      return calculate_speculative_output_stats_typed<int32_t>(
          tokens, num_speculative_tokens);
    case torch::kInt16:
      return calculate_speculative_output_stats_typed<int16_t>(
          tokens, num_speculative_tokens);
    case torch::kInt8:
      return calculate_speculative_output_stats_typed<int8_t>(
          tokens, num_speculative_tokens);
    default:
      return {};
  }
}

}  // namespace xllm::worker_service_detail

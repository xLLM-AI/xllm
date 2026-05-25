/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace xllm {

std::string readable_size(size_t bytes);

std::string summarize_int32_values(const int32_t* values,
                                   size_t size,
                                   size_t limit);

std::string summarize_int32_vector(const std::vector<int32_t>& values,
                                   size_t limit = 32);

std::string summarize_string_vector(const std::vector<std::string>& values,
                                    size_t limit = 8);

std::string summarize_int_tensor(const torch::Tensor& tensor,
                                 size_t limit = 64);

std::string summarize_accepted_tokens(const torch::Tensor& accepted_tokens,
                                      size_t max_rows = 8,
                                      size_t max_width = 8);

template <typename DecodeState>
std::string summarize_decode_states(const std::vector<DecodeState>& states,
                                    size_t limit = 16) {
  std::ostringstream oss;
  oss << "[";
  const size_t print_size = std::min(states.size(), limit);
  for (size_t i = 0; i < print_size; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    const auto& state = states[i];
    oss << "{idx=" << i << ", valid=" << state.valid
        << ", token_id=" << state.token_id
        << ", position_offset=" << state.position_offset
        << ", prev_token_id=" << state.prev_token_id
        << ", all_draft_accepted=" << state.all_draft_accepted << "}";
  }
  if (states.size() > limit) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

}  // namespace xllm

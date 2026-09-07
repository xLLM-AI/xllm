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

#include "runtime/decode_graph_bucket.h"

namespace xllm::runtime {
namespace {

constexpr int64_t kGraphTokenStep = 16;

}  // namespace

int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding) {
  if (enable_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  }
  if (num_tokens <= 2) {
    return 2;
  }
  if (num_tokens <= 4) {
    return 4;
  }
  if (num_tokens <= 8) {
    return 8;
  }

  return ((num_tokens + kGraphTokenStep - 1) / kGraphTokenStep) *
         kGraphTokenStep;
}

std::vector<int32_t> get_decode_graph_dp_token_counts(
    const std::vector<int32_t>& token_counts, int32_t graph_token_count) {
  if (token_counts.empty()) {
    return {};
  }
  std::vector<int32_t> padded_counts = token_counts;
  for (int32_t& token_count : padded_counts) {
    token_count = token_count > 0 ? graph_token_count : 1;
  }
  return padded_counts;
}

int64_t get_decode_graph_dp_layout_token_count(int32_t dp_size,
                                               int32_t graph_token_count) {
  return static_cast<int64_t>(dp_size) * graph_token_count;
}

}  // namespace xllm::runtime

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

#include <cstdint>

namespace xllm::runtime {

// Decode graph shape fields supplied by the owning Engine. For MTP,
// num_decoding_tokens is the number of token rows produced per sequence in a
// decode step; it is normally num_speculative_tokens + 1.
struct DecodeGraphExecutionShape {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
};

// Returns the padded token-row bucket shared by decode graph executors. When
// no-padding mode is enabled each exact token count is its own graph shape.
int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding);

}  // namespace xllm::runtime

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

}  // namespace xllm::runtime

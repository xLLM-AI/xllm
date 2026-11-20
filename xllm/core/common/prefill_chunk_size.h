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

#include <glog/logging.h>

#include <cstdint>

namespace xllm {

// A positive prefill chunk resolved from a CLI/API request and its batch limit.
class PrefillChunkSize final {
 public:
  PrefillChunkSize(int32_t requested_tokens, int32_t max_batch_tokens)
      : tokens_(requested_tokens < 0 ? max_batch_tokens : requested_tokens) {
    CHECK_GT(max_batch_tokens, 0) << "max_tokens_per_batch must be positive";
    CHECK_NE(requested_tokens, 0) << "max_tokens_per_chunk_for_prefill must be "
                                     "positive or negative for auto";
  }

  int32_t tokens() const { return tokens_; }

 private:
  int32_t tokens_;
};

}  // namespace xllm

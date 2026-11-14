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

#include <cstdint>

namespace xllm {

// The caller invokes this only after forward completes, before follower output
// filtering. The configured index counts requests, not positions in a batch.
inline bool should_inject_minimax_h3_failure(bool qualified_profile,
                                             bool enabled,
                                             int64_t configured_rank,
                                             int64_t configured_request,
                                             int32_t rank,
                                             int64_t request_index) {
  return qualified_profile && enabled && configured_rank >= 0 &&
         configured_rank < 16 && configured_request > 0 && request_index > 0 &&
         configured_rank == rank && configured_request == request_index;
}

}  // namespace xllm

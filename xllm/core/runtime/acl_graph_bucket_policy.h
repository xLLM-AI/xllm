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

#pragma once

#include <cstdint>

namespace xllm::npu {

inline bool is_acl_graph_warmup_batch_size(uint32_t batch_size,
                                           uint32_t max_batch_size,
                                           uint32_t dp_size) {
  if (batch_size == 0 || max_batch_size == 0 || dp_size == 0) {
    return false;
  }
  const uint32_t max_local_batch_size =
      (max_batch_size + dp_size - 1) / dp_size;
  if (batch_size > max_local_batch_size) {
    return false;
  }

  auto is_global_warmup_bucket = [max_batch_size](uint32_t global_batch_size) {
    if (global_batch_size == max_batch_size) {
      return true;
    }
    if (global_batch_size <= 16) {
      return (global_batch_size & (global_batch_size - 1)) == 0;
    }
    return global_batch_size >= 32 && global_batch_size % 16 == 0;
  };
  for (uint32_t global_batch_size = dp_size;
       global_batch_size <= max_batch_size;
       ++global_batch_size) {
    if (is_global_warmup_bucket(global_batch_size) &&
        (global_batch_size + dp_size - 1) / dp_size == batch_size) {
      return true;
    }
  }
  return false;
}

inline bool is_acl_graph_decode_capture_allowed(uint32_t batch_size,
                                                uint32_t max_batch_size,
                                                uint32_t dp_size,
                                                bool is_graph_warmup) {
  if (is_acl_graph_warmup_batch_size(batch_size, max_batch_size, dp_size)) {
    return true;
  }
  const uint32_t max_local_batch_size =
      (max_batch_size + dp_size - 1) / dp_size;
  return is_graph_warmup && batch_size <= max_local_batch_size;
}

}  // namespace xllm::npu

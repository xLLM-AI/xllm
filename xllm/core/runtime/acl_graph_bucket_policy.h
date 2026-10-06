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

namespace xllm::npu {

inline bool is_acl_graph_compatibility_batch_size(uint64_t batch_size,
                                                  uint64_t max_batch_size) {
  if (batch_size == max_batch_size) {
    return true;
  }
  if (batch_size <= 16) {
    return batch_size != 0 && (batch_size & (batch_size - 1)) == 0;
  }
  return batch_size >= 32 && batch_size % 16 == 0;
}

inline uint32_t acl_graph_max_global_batch_size(
    uint32_t configured_max_batch_size,
    uint32_t local_batch_size_limit,
    uint32_t dp_size) {
  if (configured_max_batch_size == 0 || local_batch_size_limit == 0 ||
      dp_size == 0) {
    return 0;
  }
  const uint64_t graph_capacity =
      static_cast<uint64_t>(local_batch_size_limit) * dp_size;
  return graph_capacity < configured_max_batch_size
             ? static_cast<uint32_t>(graph_capacity)
             : configured_max_batch_size;
}

inline uint32_t acl_graph_max_full_local_batch_size(uint32_t max_batch_size,
                                                    uint32_t dp_size) {
  return dp_size == 0 ? 0 : max_batch_size / dp_size;
}

inline bool is_acl_graph_warmup_batch_size(uint32_t batch_size,
                                           uint32_t max_batch_size,
                                           uint32_t dp_size,
                                           uint32_t num_decoding_tokens = 1) {
  if (batch_size == 0 || max_batch_size == 0 || dp_size == 0) {
    return false;
  }
  const uint32_t max_local_batch_size =
      (max_batch_size + dp_size - 1) / dp_size;
  if (batch_size > max_local_batch_size) {
    return false;
  }

  for (uint32_t global_batch_size = dp_size;
       global_batch_size <= max_batch_size;
       ++global_batch_size) {
    if (is_acl_graph_compatibility_batch_size(global_batch_size,
                                              max_batch_size) &&
        (global_batch_size + dp_size - 1) / dp_size == batch_size) {
      return true;
    }
  }

  // No-padding MTP warmup also schedules sequence counts whose token-row
  // counts are compatibility buckets. Admit the same counts at runtime so
  // every graph slot can capture them during warmup.
  const uint32_t max_full_local_batch_size =
      acl_graph_max_full_local_batch_size(max_batch_size, dp_size);
  if (batch_size > max_full_local_batch_size) {
    return false;
  }
  const uint64_t token_batch_size =
      static_cast<uint64_t>(batch_size) * num_decoding_tokens;
  const uint64_t max_token_batch_size =
      static_cast<uint64_t>(max_full_local_batch_size) * num_decoding_tokens;
  return num_decoding_tokens > 1 && is_acl_graph_compatibility_batch_size(
                                        token_batch_size, max_token_batch_size);
}

inline bool is_acl_graph_decode_capture_allowed(
    uint32_t batch_size,
    uint32_t max_batch_size,
    uint32_t dp_size,
    bool is_graph_warmup,
    uint32_t num_decoding_tokens = 1) {
  if (is_acl_graph_warmup_batch_size(
          batch_size, max_batch_size, dp_size, num_decoding_tokens)) {
    return true;
  }
  const uint32_t max_local_batch_size =
      (max_batch_size + dp_size - 1) / dp_size;
  return is_graph_warmup && batch_size <= max_local_batch_size;
}

}  // namespace xllm::npu

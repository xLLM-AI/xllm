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

#include "core/runtime/acl_graph_bucket_policy.h"

#include "core/framework/config/speculative_config.h"

namespace xllm::npu {

int32_t acl_graph_decode_tokens_per_sequence(
    int32_t num_speculative_tokens,
    std::string_view speculative_algorithm,
    bool is_draft_engine) {
  if (is_draft_engine) {
    if (SpeculativeConfig::is_dspark_algorithm(speculative_algorithm)) {
      return num_speculative_tokens;
    }
    if (SpeculativeConfig::is_eagle_family(speculative_algorithm)) {
      return 1;
    }
  }
  return num_speculative_tokens + 1;
}

bool is_acl_graph_decode_capture_allowed(uint32_t batch_size,
                                         uint32_t max_batch_size,
                                         uint32_t dp_size,
                                         bool is_graph_warmup,
                                         int32_t num_speculative_tokens,
                                         std::string_view speculative_algorithm,
                                         bool is_draft_engine) {
  if (batch_size == 0 || max_batch_size == 0 || dp_size == 0) {
    return false;
  }
  const uint32_t max_local_batch_size =
      (static_cast<uint64_t>(max_batch_size) + dp_size - 1) / dp_size;
  if (batch_size > max_local_batch_size) {
    return false;
  }
  if (is_graph_warmup) {
    return true;
  }
  const uint64_t first_global_batch_size = std::max<uint64_t>(
      dp_size, static_cast<uint64_t>(batch_size - 1) * dp_size + 1);
  const uint64_t last_global_batch_size = std::min<uint64_t>(
      max_batch_size, static_cast<uint64_t>(batch_size) * dp_size);
  for (uint64_t global_batch_size = first_global_batch_size;
       global_batch_size <= last_global_batch_size;
       ++global_batch_size) {
    if (is_acl_graph_compatibility_batch_size(global_batch_size,
                                              max_batch_size)) {
      return true;
    }
  }

  // No-padding MTP warmup also captures compatibility buckets in token rows.
  const uint32_t max_full_local_batch_size =
      acl_graph_max_full_local_batch_size(max_batch_size, dp_size);
  if (batch_size > max_full_local_batch_size) {
    return false;
  }
  const int32_t num_decoding_tokens = acl_graph_decode_tokens_per_sequence(
      num_speculative_tokens, speculative_algorithm, is_draft_engine);
  const uint64_t token_batch_size =
      static_cast<uint64_t>(batch_size) * num_decoding_tokens;
  const uint64_t max_token_batch_size =
      static_cast<uint64_t>(max_full_local_batch_size) * num_decoding_tokens;
  return num_decoding_tokens > 1 && is_acl_graph_compatibility_batch_size(
                                        token_batch_size, max_token_batch_size);
}

}  // namespace xllm::npu

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
#include <vector>

#include "layers/common/attention_metadata.h"

namespace xllm::layer::detail {

inline std::vector<int64_t> make_decode_actual_seq_lengths(int64_t num_tokens) {
  std::vector<int64_t> actual_seq_lengths;
  actual_seq_lengths.reserve(static_cast<size_t>(num_tokens));
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    actual_seq_lengths.emplace_back(token_idx + 1);
  }
  return actual_seq_lengths;
}

// FIA requires sparse mode 3 when an explicit attention mask is used or the
// query is causal; DFlash2 supplies a band-mode override via fia_sparse_mode,
// so the mask and causality controls stay independent of it.
inline int64_t resolve_fia_sparse_mode(const AttentionMetadata& metadata) {
  if (metadata.fia_sparse_mode >= 0) {
    return metadata.fia_sparse_mode;
  }
  return (metadata.fia_attn_mask.defined() || metadata.is_causal) ? 3 : 0;
}

}  // namespace xllm::layer::detail

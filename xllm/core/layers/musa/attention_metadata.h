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

#include <torch/torch.h>

#include "layers/common/attention_metadata.h"

namespace xllm::layer::musa {

struct AttentionMetadata final : ::xllm::layer::AttentionMetadata {
  bool initialized = false;
  torch::Tensor paged_kv_indptr_host;
  torch::Tensor paged_kv_indices_host;
  torch::Tensor paged_kv_last_page_len_host;
  bool share_fa3_scheduler_metadata = false;
  mutable torch::Tensor fa3_scheduler_metadata;
};

// MUSA producers store this derived type in the common transport pointer.
inline AttentionMetadata& get_attention_metadata(
    ::xllm::layer::AttentionMetadata& metadata) {
  return static_cast<AttentionMetadata&>(metadata);
}

inline const AttentionMetadata& get_attention_metadata(
    const ::xllm::layer::AttentionMetadata& metadata) {
  return static_cast<const AttentionMetadata&>(metadata);
}

}  // namespace xllm::layer::musa

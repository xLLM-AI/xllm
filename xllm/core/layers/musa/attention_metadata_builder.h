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

#include <memory>
#include <optional>

#include "layers/musa/attention_metadata.h"

namespace xllm {

struct ModelInputParams;

namespace layer::musa {

AttentionMetadata build_attention_metadata(
    const ModelInputParams& params,
    bool enable_mla,
    const std::optional<torch::Tensor>& attn_mask = {},
    const std::optional<torch::Device>& device = std::nullopt);

inline std::shared_ptr<AttentionMetadata> create_attention_metadata_seed(
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len) {
  auto metadata = std::make_shared<AttentionMetadata>();
  const auto to_host = [](const torch::Tensor& tensor) {
    if (!tensor.defined()) {
      return torch::Tensor();
    }
    torch::Tensor result =
        tensor.device().is_cpu() ? tensor : tensor.to(torch::kCPU);
    if (result.scalar_type() != torch::kInt32) {
      result = result.to(torch::kInt32);
    }
    return result.contiguous();
  };
  metadata->paged_kv_indptr_host = to_host(paged_kv_indptr);
  metadata->paged_kv_indices_host = to_host(paged_kv_indices);
  metadata->paged_kv_last_page_len_host = to_host(paged_kv_last_page_len);
  return metadata;
}

}  // namespace layer::musa
}  // namespace xllm

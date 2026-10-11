/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/kv_cache/layout/kv_shard_layout.h"

namespace xllm {
struct ModelArgs;
class ModelInputParams;

namespace layer {

struct AttentionMetadata;

struct AttentionMetadataBuildOptions {
  bool materialize_linear_state_validity = true;
  std::optional<KVShardLayout> kv_shard_layout;
};

// Builder class for AttentionMetadata to avoid circular dependency.
// This class handles building AttentionMetadata from ModelInputParams,
// allowing attention_metadata.h to not depend on model_input_params.h.
class AttentionMetadataBuilder {
 public:
  // One sparse decode row per token. No compact CSR or Host length mirrors:
  // page crossings change values, not the graph's allocation contract.
  static AttentionMetadata build_mtp_sparse_decode(
      const torch::Tensor& block_table,
      const torch::Tensor& kv_seq_lens,
      const torch::Tensor& slots,
      int32_t block_size);

  // Build AttentionMetadata from ModelInputParams with default compute_dtype
  // ("float").
  static AttentionMetadata build(
      const ModelInputParams& params,
      bool enable_mla,
      const std::optional<torch::Tensor>& attn_mask = {},
      const std::optional<torch::Device>& device = std::nullopt,
      const AttentionMetadataBuildOptions& build_options = {});

  // Build AttentionMetadata from ModelInputParams with specified compute_dtype.
  static AttentionMetadata build(
      const ModelInputParams& params,
      bool enable_mla,
      const std::string& compute_dtype,
      const std::optional<torch::Tensor>& attn_mask = {},
      const std::optional<torch::Device>& device = std::nullopt,
      const AttentionMetadataBuildOptions& build_options = {});
};

}  // namespace layer
}  // namespace xllm

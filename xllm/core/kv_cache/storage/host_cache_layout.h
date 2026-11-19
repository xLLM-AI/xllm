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
#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"

namespace xllm {

struct HostCacheValidationOptions {
  double host_blocks_factor = 0.0;
  int64_t device_block_count = 0;
  bool supports_host_kv_offload = false;
  bool enable_prefix_cache = true;
  bool enable_disagg_pd = false;
  bool enable_kvcache_store = false;
  InstanceRole instance_role = InstanceRole::DEFAULT;
  bool has_key_cache_shape = true;
  bool has_grouped_cache_layout = false;
  bool supports_grouped_cache_offload = false;
  bool has_conv_cache_shape = false;
  bool has_ssm_cache_shape = false;
  bool has_kpool_tail_shape = false;
  bool requires_indexer_row_folding = false;
  std::string kv_cache_dtype = "auto";
  std::string indexer_cache_dtype = "auto";
  std::string model_type;
};

// Scale a device block count to the host block count using host_blocks_factor
// (clamped to >= 1.0 so the host pool is never smaller than the device pool).
int64_t scale_host_block_count(int64_t block_count, double host_blocks_factor);

// Return an actionable error for an unsupported host prefix-cache
// configuration, or std::nullopt when the configuration is valid.
std::optional<std::string> validate_host_cache_options(
    const HostCacheValidationOptions& options);

// Build a host tensor shape from a per-layer device shape by scaling dim 0
// (block count) by host_blocks_factor.
std::vector<int64_t> build_host_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor);

// Build a grouped host tensor shape: scales dim 0 then inserts a layer
// dimension at index 1, yielding [host_blocks, layer_count, ...per_block_dims].
std::vector<int64_t> build_host_group_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor,
    int64_t layer_count);

// Group physical indexer pages into one transfer row per logical KV block.
std::vector<int64_t> fold_shape_for_host_transfer(
    const std::vector<int64_t>& shape,
    int64_t logical_blocks);

}  // namespace xllm

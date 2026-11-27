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

#include <torch/types.h>

#include <optional>
#include <string>
#include <vector>

#include "core/kv_cache/layout/paged_cache_partition.h"
#include "core/kv_cache/layout/recurrent_cache_partition.h"

namespace xllm {

class KVCacheCapacity;
class KVCacheShape;
class ModelArgs;

// Model/backend adaptation stays outside the framework-independent partition
// type. Qwen and GLM hybrids describe Full history and recurrent state
// separately. GLM KPool tails share the recurrent slot even in Full-only MTP
// drafts. Grouped attention describes each window/history ID space separately.
std::optional<PagedCachePartition> build_paged_cache_partition(
    CachePartitionId id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    const KVCacheShape& shape,
    torch::ScalarType dtype,
    int32_t token_stride,
    bool enable_prefix_cache);

std::vector<PagedCachePartition> build_grouped_cache_partitions(
    const std::string& model_id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    torch::ScalarType dtype,
    int32_t token_stride);

std::optional<RecurrentCachePartition> build_recurrent_cache_partition(
    CachePartitionId id,
    const KVCacheCapacity& capacity,
    const ModelArgs& model_args,
    const KVCacheShape& shape,
    torch::ScalarType dtype,
    int32_t checkpoint_stride,
    bool enable_prefix_cache);

}  // namespace xllm

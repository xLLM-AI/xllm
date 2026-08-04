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

#include "framework/kv_cache/kv_shard_layout.h"

namespace xllm::layer {

struct AttentionMetadata;

// Derived once for a batch and reused by every cache-sharded attention layer.
// The original logical metadata remains unchanged for consumers that need it.
struct KVShardBatchMetadata {
  torch::Tensor local_slot_mapping;
  torch::Tensor expanded_indexer_block_table;
};

torch::Tensor localize_kv_shard_slots(const torch::Tensor& logical_slots,
                                      const KVShardLayout& layout);

torch::Tensor expand_kv_shard_indexer_block_table(
    const torch::Tensor& logical_block_table,
    const KVShardLayout& layout);

std::shared_ptr<const KVShardBatchMetadata> build_kv_shard_batch_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout);

}  // namespace xllm::layer

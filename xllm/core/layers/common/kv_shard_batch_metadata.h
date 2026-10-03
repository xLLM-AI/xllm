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

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/kv_cache/kv_shard_layout.h"

namespace xllm::layer {

struct AttentionMetadata;

// Per-query metadata for running prefill selection through the rank-local
// paged-cache decode kernel. Every row represents one causal query position.
struct KVShardCausalSelectorMetadata {
  torch::Tensor block_table;
  torch::Tensor local_context_lens;
  torch::Tensor q_cu_seq_lens;
};

// Derived once for a batch and reused by every cache-sharded attention layer.
// The original logical metadata remains unchanged for consumers that need it.
struct KVShardBatchMetadata {
  torch::Tensor local_slot_mapping;
  torch::Tensor expanded_indexer_block_table;
  torch::Tensor empty_shard_mask;
  std::vector<int64_t> query_end_offsets;
  std::vector<int64_t> local_kv_lengths;
  std::vector<int64_t> local_context_lengths;
  bool has_context = false;
  int32_t kv_split_size = 1;
  int32_t kv_split_rank = 0;
};

torch::Tensor localize_kv_shard_slots(const torch::Tensor& logical_slots,
                                      const KVShardLayout& layout);

// Returns the number of tokens owned by this rank in each global causal
// prefix. The input and output use the same integer type and device.
torch::Tensor localize_kv_shard_context_lens(
    const torch::Tensor& global_context_lens,
    const KVShardLayout& layout);

torch::Tensor expand_kv_shard_indexer_block_table(
    const torch::Tensor& logical_block_table,
    const KVShardLayout& layout);

KVShardCausalSelectorMetadata build_kv_shard_causal_selector_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout);

struct KVShardBatchMetadataBuildOptions {
  bool materialize_indexer_block_table = true;
  bool materialize_attention_lengths = false;
};

std::shared_ptr<const KVShardBatchMetadata> build_kv_shard_batch_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout,
    const KVShardBatchMetadataBuildOptions& build_options = {});

}  // namespace xllm::layer

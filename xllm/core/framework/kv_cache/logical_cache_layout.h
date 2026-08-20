/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <string>
#include <vector>

namespace xllm {

enum class LogicalShardKind : int8_t {
  REPLICATED = 0,
  SHARDED = 1,
  COMPOSITE = 2,
};

enum class CacheResourceScope : int8_t {
  BLOCK = 0,
  SEQUENCE = 1,
};

// Describes a compact mapping from canonical logical bytes to bytes within one
// cache resource (one block or one sequence slot). A repeated span represents
// repeat_count equally-sized regions with independent logical and physical
// strides. The planner expands these immutable patterns only while linking.
struct LogicalSpan {
  std::string logical_tensor;
  uint64_t logical_offset_bytes = 0;
  uint64_t physical_offset_bytes = 0;
  uint64_t bytes_per_region = 0;
  uint64_t repeat_count = 1;
  uint64_t logical_stride_bytes = 0;
  uint64_t physical_stride_bytes = 0;
  // Static writer for this span among source TP replicas. For a genuinely
  // sharded span this is the rank that owns the shard; for replicated data it
  // is the deterministic lowest equivalent rank.
  int32_t owner_tp_rank = 0;
};

struct LogicalShardDescriptor {
  LogicalShardKind kind = LogicalShardKind::REPLICATED;
  CacheResourceScope resource_scope = CacheResourceScope::BLOCK;
  std::vector<LogicalSpan> spans;
};

// Producer-side semantic context used to construct descriptors. It is filled
// from the model and parallel layout before cache registration; the transfer
// planner never infers these values from a tensor role.
struct CacheTensorLayoutContext {
  int32_t tp_rank = 0;
  int32_t tp_size = 1;
  int64_t block_token_capacity = 0;
  int64_t kv_head_count = 0;
  int64_t index_head_count = 0;
  int64_t linear_key_head_count = 0;
  int64_t linear_value_head_count = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_ssm_checkpoint_stride = 1;
  bool enable_mla = false;
  bool head_major_layout = false;
};

}  // namespace xllm

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

#include "framework/kv_cache/cache_layout_builder.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace xllm {

namespace {

bool fail(const std::string& message, std::string* error) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool get_head_placement(int64_t global_head_count,
                        int64_t local_head_count,
                        int32_t tp_rank,
                        int32_t tp_size,
                        int64_t* first_global_head,
                        int32_t* replica_count,
                        std::string* error) {
  if (global_head_count <= 0 || local_head_count <= 0 || tp_size <= 0 ||
      tp_rank < 0 || tp_rank >= tp_size || first_global_head == nullptr ||
      replica_count == nullptr) {
    return fail("invalid TP head placement", error);
  }

  if (global_head_count >= tp_size) {
    if (global_head_count % tp_size != 0 ||
        local_head_count != global_head_count / tp_size) {
      return fail("global head count is not evenly sharded by TP", error);
    }
    *first_global_head = static_cast<int64_t>(tp_rank) * local_head_count;
    *replica_count = 1;
    return true;
  }

  if (tp_size % global_head_count != 0 || local_head_count != 1) {
    return fail("KV-head replication requires TP to be a multiple of heads",
                error);
  }
  *replica_count = tp_size / static_cast<int32_t>(global_head_count);
  *first_global_head = tp_rank / *replica_count;
  return true;
}

uint64_t tensor_stride_bytes(const torch::Tensor& tensor, int64_t dim) {
  return static_cast<uint64_t>(tensor.stride(dim)) * tensor.element_size();
}

bool describe_replicated_tensor(KVCacheTensor* cache_tensor,
                                std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() == 0 || tensor.size(0) <= 0) {
    return fail("replicated cache tensor must be contiguous and non-empty",
                error);
  }

  const uint64_t resource_count = static_cast<uint64_t>(tensor.size(0));
  const uint64_t tensor_bytes = static_cast<uint64_t>(tensor.nbytes());
  if (tensor_bytes % resource_count != 0) {
    return fail("cache tensor bytes are not divisible by resource count",
                error);
  }

  LogicalSpan span;
  span.logical_tensor = cache_tensor->role.to_string();
  span.bytes_per_region = tensor_bytes / resource_count;
  span.owner_tp_rank = 0;

  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::REPLICATED;
  descriptor.resource_scope = cache_tensor->sequence_scoped
                                  ? CacheResourceScope::SEQUENCE
                                  : CacheResourceScope::BLOCK;
  descriptor.spans.emplace_back(std::move(span));
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool describe_attention_heads(const CacheTensorLayoutContext& context,
                              int64_t global_head_count,
                              KVCacheTensor* cache_tensor,
                              std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() < 3 || tensor.dim() > 4) {
    return fail("head-sharded cache tensor must be contiguous and 3D/4D",
                error);
  }

  const int64_t head_axis = context.head_major_layout ? 1 : 2;
  const int64_t token_axis = context.head_major_layout ? 2 : 1;
  if (head_axis >= tensor.dim() || token_axis >= tensor.dim()) {
    return fail("cache tensor does not contain head and token axes", error);
  }
  if (context.block_token_capacity > 0 &&
      tensor.size(token_axis) != context.block_token_capacity) {
    return fail("cache tensor token capacity differs from producer metadata",
                error);
  }

  const int64_t local_head_count = tensor.size(head_axis);
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(global_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  uint64_t bytes_per_head = tensor.element_size();
  const int64_t inner_begin = std::max(head_axis, token_axis) + 1;
  for (int64_t dim = inner_begin; dim < tensor.dim(); ++dim) {
    uint64_t next_bytes = 0;
    if (!checked_multiply(bytes_per_head,
                          static_cast<uint64_t>(tensor.size(dim)),
                          &next_bytes)) {
      return fail("cache tensor head byte size overflow", error);
    }
    bytes_per_head = next_bytes;
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind =
      replicated ? LogicalShardKind::REPLICATED : LogicalShardKind::SHARDED;
  descriptor.resource_scope = CacheResourceScope::BLOCK;
  descriptor.spans.reserve(static_cast<size_t>(local_head_count));
  const uint64_t logical_token_stride =
      static_cast<uint64_t>(global_head_count) * bytes_per_head;
  const uint64_t physical_token_stride =
      tensor_stride_bytes(tensor, token_axis);
  const uint64_t physical_head_stride = tensor_stride_bytes(tensor, head_axis);
  const uint64_t token_count = static_cast<uint64_t>(tensor.size(token_axis));

  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = cache_tensor->role.to_string();
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        static_cast<uint64_t>(local_head) * physical_head_stride;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = token_count;
    span.logical_stride_bytes = logical_token_stride;
    span.physical_stride_bytes = physical_token_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor.spans.emplace_back(std::move(span));
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool describe_ssm(const CacheTensorLayoutContext& context,
                  KVCacheTensor* cache_tensor,
                  std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() != 4 ||
      context.linear_ssm_checkpoint_stride <= 0 ||
      tensor.size(0) % context.linear_ssm_checkpoint_stride != 0) {
    return fail("SSM cache tensor must use [slot, head, key, value]", error);
  }

  const int64_t local_head_count = tensor.size(1);
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(context.linear_value_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  uint64_t bytes_per_head = tensor.element_size();
  for (int64_t dim = 2; dim < tensor.dim(); ++dim) {
    uint64_t next_bytes = 0;
    if (!checked_multiply(bytes_per_head,
                          static_cast<uint64_t>(tensor.size(dim)),
                          &next_bytes)) {
      return fail("SSM cache head byte size overflow", error);
    }
    bytes_per_head = next_bytes;
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind =
      replicated ? LogicalShardKind::REPLICATED : LogicalShardKind::SHARDED;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.reserve(static_cast<size_t>(local_head_count));
  const uint64_t checkpoint_stride =
      static_cast<uint64_t>(context.linear_ssm_checkpoint_stride);
  const uint64_t logical_checkpoint_stride =
      static_cast<uint64_t>(context.linear_value_head_count) * bytes_per_head;
  const uint64_t physical_checkpoint_stride = tensor_stride_bytes(tensor, 0);
  const uint64_t physical_head_stride = tensor_stride_bytes(tensor, 1);
  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = cache_tensor->role.to_string();
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        static_cast<uint64_t>(local_head) * physical_head_stride;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = checkpoint_stride;
    span.logical_stride_bytes = logical_checkpoint_stride;
    span.physical_stride_bytes = physical_checkpoint_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor.spans.emplace_back(std::move(span));
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool append_conv_component(const CacheTensorLayoutContext& context,
                           const torch::Tensor& tensor,
                           const std::string& logical_tensor,
                           int64_t global_head_count,
                           int64_t local_head_count,
                           uint64_t component_offset_bytes,
                           LogicalShardDescriptor* descriptor,
                           std::string* error) {
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(global_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  const uint64_t bytes_per_head =
      static_cast<uint64_t>(context.linear_key_head_dim) *
      tensor.element_size();
  const uint64_t state_count = static_cast<uint64_t>(tensor.size(1));
  const uint64_t physical_state_stride = tensor_stride_bytes(tensor, 1);
  const uint64_t logical_state_stride =
      static_cast<uint64_t>(global_head_count) * bytes_per_head;
  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = logical_tensor;
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        component_offset_bytes +
        static_cast<uint64_t>(local_head) * bytes_per_head;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = state_count;
    span.logical_stride_bytes = logical_state_stride;
    span.physical_stride_bytes = physical_state_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor->spans.emplace_back(std::move(span));
  }
  return true;
}

bool describe_conv(const CacheTensorLayoutContext& context,
                   KVCacheTensor* cache_tensor,
                   std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() != 3 ||
      context.linear_key_head_dim <= 0) {
    return fail("CONV cache tensor must use [slot, state, packed_heads]",
                error);
  }

  const int64_t local_key_heads =
      context.linear_key_head_count >= context.tp_size
          ? context.linear_key_head_count / context.tp_size
          : 1;
  const int64_t local_value_heads =
      context.linear_value_head_count >= context.tp_size
          ? context.linear_value_head_count / context.tp_size
          : 1;
  const int64_t expected_features =
      context.linear_key_head_dim * (local_key_heads * 2 + local_value_heads);
  if (tensor.size(2) != expected_features) {
    return fail("CONV packed feature size differs from producer metadata",
                error);
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::COMPOSITE;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.reserve(
      static_cast<size_t>(local_key_heads * 2 + local_value_heads));
  const uint64_t bytes_per_head =
      static_cast<uint64_t>(context.linear_key_head_dim) *
      tensor.element_size();
  if (!append_conv_component(context,
                             tensor,
                             "conv_key_a",
                             context.linear_key_head_count,
                             local_key_heads,
                             /*component_offset_bytes=*/0,
                             &descriptor,
                             error) ||
      !append_conv_component(
          context,
          tensor,
          "conv_key_b",
          context.linear_key_head_count,
          local_key_heads,
          static_cast<uint64_t>(local_key_heads) * bytes_per_head,
          &descriptor,
          error) ||
      !append_conv_component(
          context,
          tensor,
          "conv_value",
          context.linear_value_head_count,
          local_value_heads,
          static_cast<uint64_t>(local_key_heads * 2) * bytes_per_head,
          &descriptor,
          error)) {
    return false;
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool is_kv_head_role(KVCacheTensorRole role) {
  return role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
         role == KVCacheTensorRole::KEY_SCALE ||
         role == KVCacheTensorRole::VALUE_SCALE ||
         role == KVCacheTensorRole::CACHE_SCALE;
}

}  // namespace

bool describe_cache_tensor(const CacheTensorLayoutContext& context,
                           KVCacheTensor* cache_tensor,
                           std::string* error) {
  if (cache_tensor == nullptr || !cache_tensor->tensor.defined() ||
      cache_tensor->tensor.numel() <= 0) {
    return fail("cache tensor must be defined and non-empty", error);
  }
  if (cache_tensor->shard_descriptor.has_value()) {
    return true;
  }

  // MLA caches are logical replicas even when physical cache formats differ
  // from ordinary K/V layouts. The deterministic TP owner prevents duplicate
  // writes while every destination replica is still populated.
  if (context.enable_mla) {
    return describe_replicated_tensor(cache_tensor, error);
  }
  if (cache_tensor->role == KVCacheTensorRole::CONV &&
      context.linear_key_head_count > 0 &&
      context.linear_value_head_count > 0) {
    return describe_conv(context, cache_tensor, error);
  }
  if (cache_tensor->role == KVCacheTensorRole::SSM &&
      context.linear_value_head_count > 0) {
    return describe_ssm(context, cache_tensor, error);
  }
  if (is_kv_head_role(cache_tensor->role) && context.kv_head_count > 0) {
    return describe_attention_heads(
        context, context.kv_head_count, cache_tensor, error);
  }
  if (cache_tensor->role == KVCacheTensorRole::INDEX ||
      cache_tensor->role == KVCacheTensorRole::INDEX_SCALE) {
    // The lightning indexer query uses index_head_count heads, but its cached
    // key is a single shared logical head. KVCacheShape and the DeepSeek V4
    // grouped cache both allocate INDEX/INDEX_SCALE with a physical head
    // dimension of one on every TP rank.
    return describe_attention_heads(
        context, /*global_head_count=*/1, cache_tensor, error);
  }

  // Roles whose producer exposes no head axis are explicit whole-resource
  // replicas. This covers grouped state tensors and preserves their independent
  // logical identities instead of treating them as KEY-shaped bytes.
  return describe_replicated_tensor(cache_tensor, error);
}

}  // namespace xllm

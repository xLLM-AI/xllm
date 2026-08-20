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

#include <gtest/gtest.h>

#include <string>

namespace xllm {

namespace {

TEST(CacheLayoutBuilderTest, DescribesTokenMajorGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 3;
  context.block_token_capacity = 3;
  context.kv_head_count = 6;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 2, 4}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::BLOCK);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[1].logical_offset_bytes, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 6U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 1);
}

TEST(CacheLayoutBuilderTest, SelectsStableOwnerForMqaReplicas) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 1;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, SelectsContiguousReplicaGroupForGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 2;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 8U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 2);
}

TEST(CacheLayoutBuilderTest, DescribesMlaAsWholeResourceReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.enable_mla = true;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 16}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region,
            static_cast<uint64_t>(tensor.tensor.nbytes() / 2));
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerKeyAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 7;
  context.tp_size = 8;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX,
                       torch::zeros({2, 3, 1, 4}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerScaleAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX_SCALE,
                       torch::zeros({2, 3, 1}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSequenceScopedSsmHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({2, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 1U);
}

TEST(CacheLayoutBuilderTest, DescribesCheckpointedSsmRowsPerLogicalSlot) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  context.linear_ssm_checkpoint_stride = 3;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({6, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 4U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 3U * 4U * 4U);
}

TEST(CacheLayoutBuilderTest, DescribesCompositeConvState) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_key_head_count = 4;
  context.linear_value_head_count = 2;
  context.linear_key_head_dim = 3;
  KVCacheTensor tensor{KVCacheTensorRole::CONV,
                       torch::zeros({2, 5, 15}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::COMPOSITE);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 5U);
  EXPECT_EQ(descriptor.spans[0].logical_tensor, "conv_key_a");
  EXPECT_EQ(descriptor.spans[2].logical_tensor, "conv_key_b");
  EXPECT_EQ(descriptor.spans[4].logical_tensor, "conv_value");
  EXPECT_EQ(descriptor.spans[0].repeat_count, 5U);
}

}  // namespace

}  // namespace xllm

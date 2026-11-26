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

#include "core/kv_cache/layout/recurrent_cache_partition.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(RecurrentCachePartitionTest, RejectsInvalidStateStorage) {
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/8,
                                        /*ssm_snapshots_per_slot=*/0,
                                        /*enable_prefix_cache=*/true,
                                        {0},
                                        CacheScalarType::BFLOAT16,
                                        CacheScalarType::FLOAT32)),
               "must contain SSM snapshots");
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/0,
                                        /*ssm_snapshots_per_slot=*/1,
                                        /*enable_prefix_cache=*/true,
                                        {0},
                                        CacheScalarType::BFLOAT16,
                                        CacheScalarType::FLOAT32)),
               "checkpoint stride must be positive");
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/8,
                                        /*ssm_snapshots_per_slot=*/1,
                                        /*enable_prefix_cache=*/true,
                                        {},
                                        CacheScalarType::BFLOAT16,
                                        CacheScalarType::FLOAT32)),
               "must own layers");
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/8,
                                        /*ssm_snapshots_per_slot=*/1,
                                        /*enable_prefix_cache=*/true,
                                        {1, 1},
                                        CacheScalarType::BFLOAT16,
                                        CacheScalarType::FLOAT32)),
               "strictly increasing");
}

}  // namespace
}  // namespace xllm

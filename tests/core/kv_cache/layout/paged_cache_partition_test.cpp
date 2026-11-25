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

#include "core/kv_cache/layout/paged_cache_partition.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(PagedCachePartitionTest, RejectsAmbiguousLayerOwnership) {
  EXPECT_DEATH(
      (PagedCachePartition(
          CachePartitionId{},
          /*token_stride=*/4,
          /*enable_prefix_cache=*/true,
          {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0, 0}}})),
      "strictly increasing");
  EXPECT_DEATH(
      (PagedCachePartition(
          CachePartitionId{},
          /*token_stride=*/4,
          /*enable_prefix_cache=*/true,
          {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {1, 0}}})),
      "strictly increasing");
  EXPECT_DEATH((PagedCachePartition(
                   CachePartitionId{},
                   /*token_stride=*/4,
                   /*enable_prefix_cache=*/true,
                   {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {}}})),
               "must own layers");
  EXPECT_DEATH(
      (PagedCachePartition(
          CachePartitionId{},
          /*token_stride=*/4,
          /*enable_prefix_cache=*/true,
          {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0}},
           {CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {1}}})),
      "Duplicate cache component");
}

TEST(PagedCachePartitionTest, RejectsPartitionsWithoutPagedStorage) {
  EXPECT_DEATH((PagedCachePartition(CachePartitionId{},
                                    /*token_stride=*/4,
                                    /*enable_prefix_cache=*/true,
                                    {})),
               "must contain components");
  EXPECT_DEATH(
      (PagedCachePartition(
          CachePartitionId{},
          /*token_stride=*/0,
          /*enable_prefix_cache=*/true,
          {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0}}})),
      "token stride must be positive");
}

}  // namespace
}  // namespace xllm

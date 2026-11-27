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
  const std::vector<CacheComponent> components = {
      {CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {0}},
      {CacheComponentRole::SSM, CacheScalarType::FLOAT32, {0}}};
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/8,
                                        /*ssm_snapshots_per_slot=*/0,
                                        /*enable_prefix_cache=*/true,
                                        components)),
               "must contain SSM snapshots");
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{},
                                        /*checkpoint_stride=*/0,
                                        /*ssm_snapshots_per_slot=*/1,
                                        /*enable_prefix_cache=*/true,
                                        components)),
               "checkpoint stride must be positive");
  EXPECT_DEATH((RecurrentCachePartition(CachePartitionId{}, 8, 1, true, {})),
               "must contain components");
  EXPECT_DEATH((RecurrentCachePartition(
                   CachePartitionId{},
                   8,
                   1,
                   true,
                   {{CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {}},
                    {CacheComponentRole::SSM, CacheScalarType::FLOAT32, {}}})),
               "must own layers");
  EXPECT_DEATH(
      (RecurrentCachePartition(
          CachePartitionId{},
          8,
          1,
          true,
          {{CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {0, 0}},
           {CacheComponentRole::SSM, CacheScalarType::FLOAT32, {0, 0}}})),
      "strictly increasing");
}

TEST(RecurrentCachePartitionTest, RejectsIncompleteOrAmbiguousComponents) {
  EXPECT_DEATH(
      (RecurrentCachePartition(
          CachePartitionId{},
          8,
          1,
          true,
          {{CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {0}}})),
      "both Conv and SSM");
  EXPECT_DEATH((RecurrentCachePartition(
                   CachePartitionId{},
                   8,
                   1,
                   true,
                   {{CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {0}},
                    {CacheComponentRole::SSM, CacheScalarType::FLOAT32, {1}}})),
               "must own the same layers");
  EXPECT_DEATH(
      (RecurrentCachePartition(
          CachePartitionId{},
          8,
          0,
          true,
          {{CacheComponentRole::KPOOL_TAIL, CacheScalarType::BFLOAT16, {1}},
           {CacheComponentRole::KPOOL_TAIL, CacheScalarType::BFLOAT16, {2}}},
          /*tail_tokens_per_slot=*/7)),
      "Duplicate cache component");
  EXPECT_DEATH(
      (RecurrentCachePartition(
          CachePartitionId{},
          8,
          0,
          true,
          {{CacheComponentRole::INDEX, CacheScalarType::BFLOAT16, {1}}})),
      "Unsupported recurrent cache component");
}

TEST(RecurrentCachePartitionTest, TailRequiresIndependentTokenCapacity) {
  const std::vector<CacheComponent> components = {
      {CacheComponentRole::KPOOL_TAIL, CacheScalarType::BFLOAT16, {1}}};
  EXPECT_DEATH(
      (RecurrentCachePartition(CachePartitionId{}, 8, 1, true, components, 7)),
      "must not reserve SSM snapshots");
  EXPECT_DEATH(
      (RecurrentCachePartition(CachePartitionId{}, 8, 0, true, components, 0)),
      "tail must contain tokens");
  EXPECT_DEATH((RecurrentCachePartition(
                   CachePartitionId{},
                   8,
                   1,
                   true,
                   {{CacheComponentRole::CONV, CacheScalarType::BFLOAT16, {0}},
                    {CacheComponentRole::SSM, CacheScalarType::FLOAT32, {0}}},
                   7)),
               "Tail tokens require a KPool tail component");
}

}  // namespace
}  // namespace xllm

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace xllm::layer::mlu_v4_cp {
namespace {

TEST(DeepseekV4CpGeometryTest,
     ConvertsMluCumulativeLengthsForConcurrentRequests) {
  EXPECT_EQ(query_lengths_from_cumulative({0, 128, 256, 384}, 384),
            std::vector<int32_t>({128, 128, 128}));
  EXPECT_EQ(query_lengths_from_cumulative({0, 17, 82, 91}, 91),
            std::vector<int32_t>({17, 65, 9}));
}

TEST(DeepseekV4CpGeometryTest, RequiresEnoughTokensForTwoSidedSplit) {
  EXPECT_FALSE(should_enable_zigzag_cp({3, 2}, /*cp_size=*/2));
  EXPECT_TRUE(should_enable_zigzag_cp({4, 1}, /*cp_size=*/2));
  EXPECT_FALSE(should_enable_zigzag_cp({}, /*cp_size=*/2));
  EXPECT_FALSE(should_enable_zigzag_cp({8}, /*cp_size=*/1));
}

TEST(DeepseekV4CpGeometryTest, AssignsSymmetricFrontAndBackBuckets) {
  const DeepseekV4CpGeometry geometry = build_cp_geometry(/*cp_size=*/2, {8});

  ASSERT_EQ(geometry.rows_by_rank.size(), 2u);
  EXPECT_EQ(geometry.rows_by_rank[0], std::vector<int64_t>({0, 1, 6, 7}));
  EXPECT_EQ(geometry.rows_by_rank[1], std::vector<int64_t>({2, 3, 4, 5}));
  EXPECT_EQ(geometry.front_seq_lens_by_rank[0], std::vector<int32_t>({2}));
  EXPECT_EQ(geometry.back_seq_lens_by_rank[0], std::vector<int32_t>({2}));
  EXPECT_EQ(geometry.restore_indices,
            std::vector<int64_t>({0, 1, 4, 5, 6, 7, 2, 3}));
}

TEST(DeepseekV4CpGeometryTest, PadsAnEntirelyEmptyRegionWithOneFakeRow) {
  const DeepseekV4CpGeometry geometry = build_cp_geometry(/*cp_size=*/2, {5});

  EXPECT_EQ(geometry.rows_by_rank[0], std::vector<int64_t>({0, 1, 0}));
  EXPECT_EQ(geometry.scatter_rows_by_rank[0], std::vector<int64_t>({0, 1, 5}));
  EXPECT_EQ(geometry.front_seq_lens_by_rank[0], std::vector<int32_t>({2}));
  EXPECT_EQ(geometry.back_seq_lens_by_rank[0], std::vector<int32_t>({1}));
  EXPECT_EQ(geometry.tokens_per_rank, std::vector<int32_t>({3, 3}));
  EXPECT_EQ(geometry.restore_indices, std::vector<int64_t>({0, 1, 3, 4, 5}));
}

TEST(DeepseekV4CpGeometryTest, FakeRowsNeverOverwriteARealToken) {
  const DeepseekV4CpGeometry geometry =
      build_cp_geometry(/*cp_size=*/4, {2, 2, 8});

  for (size_t rank = 0; rank < geometry.rows_by_rank.size(); ++rank) {
    ASSERT_EQ(geometry.rows_by_rank[rank].size(),
              geometry.scatter_rows_by_rank[rank].size());
    for (size_t row = 0; row < geometry.rows_by_rank[rank].size(); ++row) {
      const int64_t gather_row = geometry.rows_by_rank[rank][row];
      const int64_t scatter_row = geometry.scatter_rows_by_rank[rank][row];
      EXPECT_TRUE(scatter_row == gather_row ||
                  scatter_row == geometry.total_tokens);
    }
  }
}

TEST(DeepseekV4CpGeometryTest, CoversEveryGlobalRowExactlyOnce) {
  const std::vector<int32_t> q_seq_lens = {9, 4, 7};
  const DeepseekV4CpGeometry geometry =
      build_cp_geometry(/*cp_size=*/3, q_seq_lens);
  const int64_t total_tokens =
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0});

  std::vector<int64_t> actual_rows;
  for (const std::vector<int64_t>& scatter_rows :
       geometry.scatter_rows_by_rank) {
    for (int64_t row : scatter_rows) {
      if (row < total_tokens) {
        actual_rows.emplace_back(row);
      }
    }
  }
  std::sort(actual_rows.begin(), actual_rows.end());
  ASSERT_EQ(static_cast<int64_t>(actual_rows.size()), total_tokens);
  for (int64_t row = 0; row < total_tokens; ++row) {
    EXPECT_EQ(actual_rows[static_cast<size_t>(row)], row);
  }
}

TEST(DeepseekV4CpContextTest, ShardsOnlyTheLeadingTokenDimension) {
  DeepseekV4CpContext context;
  context.local_row_indices = torch::tensor({1, 3}, torch::kInt64);
  const torch::Tensor global = torch::arange(24).reshape({4, 2, 3});

  const torch::Tensor local = shard_rows(global, context);

  ASSERT_EQ(local.sizes(), c10::IntArrayRef({2, 2, 3}));
  EXPECT_TRUE(torch::equal(local[0], global[1]));
  EXPECT_TRUE(torch::equal(local[1], global[3]));
}

}  // namespace
}  // namespace xllm::layer::mlu_v4_cp

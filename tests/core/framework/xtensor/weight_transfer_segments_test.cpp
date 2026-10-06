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

#include "core/framework/xtensor/weight_transfer_segments.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace xllm {
namespace {

TEST(WeightTransferSegmentsTest, FragmentedTransferPreservesLogicalWeights) {
  constexpr uint64_t kPageSize = 4;
  const std::vector<int64_t> page_ids = {9, 7, 8, 3};
  const std::vector<uint8_t> logical_weights = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint8_t> physical_weights(10 * kPageSize, 0);
  for (size_t i = 0; i < logical_weights.size(); ++i) {
    uint64_t physical_offset =
        static_cast<uint64_t>(page_ids[i / kPageSize]) * kPageSize +
        i % kPageSize;
    physical_weights[physical_offset] = logical_weights[i];
  }

  auto segments = make_weight_transfer_segments(page_ids, kPageSize);
  const std::vector<std::pair<uint64_t, uint64_t>> expected_segments = {
      {9 * kPageSize, kPageSize},
      {7 * kPageSize, 2 * kPageSize},
      {3 * kPageSize, kPageSize}};
  EXPECT_EQ(segments, expected_segments);

  std::vector<uint8_t> transferred_weights;
  transferred_weights.reserve(logical_weights.size());
  for (const auto& [offset, size] : segments) {
    transferred_weights.insert(transferred_weights.end(),
                               physical_weights.begin() + offset,
                               physical_weights.begin() + offset + size);
  }
  EXPECT_EQ(transferred_weights, logical_weights);
}

TEST(WeightTransferSegmentsTest, MergesForwardAdjacentPages) {
  const std::vector<std::pair<uint64_t, uint64_t>> expected = {{20, 30}};
  EXPECT_EQ(make_weight_transfer_segments({2, 3, 4}, /*page_size=*/10),
            expected);
}

TEST(WeightTransferSegmentsTest, KeepsDescendingAdjacentPagesSeparate) {
  const std::vector<std::pair<uint64_t, uint64_t>> expected = {
      {40, 10}, {30, 10}, {20, 10}};
  EXPECT_EQ(make_weight_transfer_segments({4, 3, 2}, /*page_size=*/10),
            expected);
}

TEST(WeightTransferSegmentsTest, EmptyPagesHaveNoTransferSegments) {
  EXPECT_TRUE(make_weight_transfer_segments({}, /*page_size=*/10).empty());
}

TEST(WeightTransferSegmentsTest, AcceptsLargestRepresentablePageEnd) {
  const uint64_t page_size = std::numeric_limits<uint64_t>::max();
  const std::vector<std::pair<uint64_t, uint64_t>> expected = {{0, page_size}};
  EXPECT_EQ(make_weight_transfer_segments({0}, page_size), expected);
}

TEST(WeightTransferSegmentsDeathTest, RejectsInvalidPageGeometry) {
  EXPECT_DEATH(make_weight_transfer_segments({-1}, /*page_size=*/4), "page_id");
  EXPECT_DEATH(make_weight_transfer_segments({0}, /*page_size=*/0),
               "page_size");
  EXPECT_DEATH(
      make_weight_transfer_segments({1}, std::numeric_limits<uint64_t>::max()),
      "page_id");
}

}  // namespace
}  // namespace xllm

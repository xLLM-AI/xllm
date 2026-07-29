/* Copyright 2025-2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace parallel_state {
namespace {

// Re-derive the CP rank of a global rank from the documented layout:
//   rank = dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size +
//   tp_rank
// where attn_tp_size = world_size / (dp_size * cp_size).
int32_t expected_cp_rank(int32_t global_rank,
                         int32_t world_size,
                         int32_t dp_size,
                         int32_t cp_size) {
  const int32_t attn_tp_size = world_size / (dp_size * cp_size);
  return (global_rank % (cp_size * attn_tp_size)) / attn_tp_size;
}

TEST(ComputeCpGroupRanks, CpSizeTwoTpFourDpOne) {
  const int32_t world_size = 8;
  const int32_t dp_size = 1;
  const int32_t cp_size = 2;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    ASSERT_EQ(ranks.size(), cp_size);
    // CP group spans ranks that differ only in the CP dimension: same dp_rank
    // and tp_rank, varying cp_rank.
    const int32_t attn_tp_size = world_size / (dp_size * cp_size);
    const int32_t tp_rank = rank % attn_tp_size;
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
      EXPECT_EQ(ranks[cp_rank],
                dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size +
                    tp_rank);
    }
    // The rank's own position in the vector is its CP rank.
    EXPECT_EQ(ranks[expected_cp_rank(rank, world_size, dp_size, cp_size)],
              rank);
  }
}

TEST(ComputeCpGroupRanks, CpSizeFourTpTwoDpTwo) {
  const int32_t world_size = 16;
  const int32_t dp_size = 2;
  const int32_t cp_size = 4;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    ASSERT_EQ(ranks.size(), cp_size);
    EXPECT_EQ(ranks[expected_cp_rank(rank, world_size, dp_size, cp_size)],
              rank);
    // Every member shares the same dp_rank and tp_rank as `rank`.
    const int32_t attn_tp_size = world_size / (dp_size * cp_size);
    const int32_t tp_rank = rank % attn_tp_size;
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    for (int32_t r : ranks) {
      EXPECT_EQ(r % attn_tp_size, tp_rank);
      EXPECT_EQ(r / (cp_size * attn_tp_size), dp_rank);
    }
  }
}

TEST(ComputeCpGroupRanks, GroupsPartitionWorldAndAreOrthogonalToTp) {
  const int32_t world_size = 16;
  const int32_t dp_size = 2;
  const int32_t cp_size = 4;
  const int32_t attn_tp_size = world_size / (dp_size * cp_size);

  // Every global rank must appear in exactly one CP group; collect all groups
  // via rank 0's perspective is insufficient, so iterate all ranks and verify
  // that two ranks share a CP group iff they share (dp_rank, tp_rank).
  auto same_cp_group = [&](int32_t a, int32_t b) {
    return compute_cp_group_ranks(a, world_size, dp_size, cp_size) ==
           compute_cp_group_ranks(b, world_size, dp_size, cp_size);
  };

  for (int32_t a = 0; a < world_size; ++a) {
    // Capture the group once: computing begin()/end() on two separate
    // temporaries would yield iterators into different containers (undefined
    // behavior).
    const std::vector<int32_t> a_group =
        compute_cp_group_ranks(a, world_size, dp_size, cp_size);
    std::set<int32_t> group_members(a_group.begin(), a_group.end());
    EXPECT_EQ(group_members.size(), cp_size);
    for (int32_t b = 0; b < world_size; ++b) {
      const bool same_dp =
          a / (cp_size * attn_tp_size) == b / (cp_size * attn_tp_size);
      const bool same_tp = a % attn_tp_size == b % attn_tp_size;
      EXPECT_EQ(same_cp_group(a, b), same_dp && same_tp);
    }
  }

  // Orthogonality to TP: a rank's TP group (same dp_rank, same cp_rank, varying
  // tp_rank) must intersect its CP group only at the rank itself.
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const std::vector<int32_t> cp_ranks =
        compute_cp_group_ranks(rank, world_size, dp_size, cp_size);
    const int32_t dp_rank = rank / (cp_size * attn_tp_size);
    const int32_t cp_rank =
        expected_cp_rank(rank, world_size, dp_size, cp_size);
    std::set<int32_t> cp_set(cp_ranks.begin(), cp_ranks.end());
    for (int32_t tp_rank = 0; tp_rank < attn_tp_size; ++tp_rank) {
      const int32_t tp_peer =
          dp_rank * (cp_size * attn_tp_size) + cp_rank * attn_tp_size + tp_rank;
      if (tp_peer == rank) {
        EXPECT_NE(cp_set.find(tp_peer), cp_set.end());
      } else {
        EXPECT_EQ(cp_set.find(tp_peer), cp_set.end());
      }
    }
  }
}

TEST(ComputeCpGroupRanks, RejectsNonIntegralAttnTpSize) {
  // world_size=8, dp_size=2, cp_size=3 => 8 not divisible by 6.
  EXPECT_DEATH(
      compute_cp_group_ranks(0, /*world_size=*/8, /*dp_size=*/2, /*cp_size=*/3),
      "");
}

}  // namespace
}  // namespace parallel_state
}  // namespace xllm

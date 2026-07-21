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

#include "framework/kv_cache_transfer/push_route.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {

TEST(PushRouteTest, SrcTpLessThanDstTpKeepsMulticast) {
  EXPECT_FALSE(use_push_owner(2, 8));

  const std::vector<int32_t> dst_ranks = get_dst_ranks(1, 2, 8, 3);
  const std::vector<int32_t> expect_ranks = {25, 27, 29, 31};
  EXPECT_EQ(dst_ranks, expect_ranks);
}

TEST(PushRouteTest, InvalidTpSizeNotUseOwnerAndReturnEmpty) {
  EXPECT_FALSE(use_push_owner(0, 4));
  EXPECT_FALSE(use_push_owner(4, 0));
  EXPECT_FALSE(use_push_owner(-1, 2));
  EXPECT_FALSE(use_push_owner(2, -1));

  const std::vector<int32_t> dst_ranks = get_dst_ranks(0, 0, 4, 0);
  EXPECT_TRUE(dst_ranks.empty());
}

TEST(PushRouteTest, SrcTpEqualsDstTpNotUseOwner) {
  EXPECT_FALSE(use_push_owner(4, 4));

  const std::vector<int32_t> dst_ranks = get_dst_ranks(2, 4, 4, 1);
  const std::vector<int32_t> expect_ranks = {6};
  EXPECT_EQ(dst_ranks, expect_ranks);
}

TEST(PushRouteTest, SrcTpGreaterThanDstTpUsesOwnerRouting) {
  EXPECT_TRUE(use_push_owner(8, 3));

  const std::vector<int32_t> owner_ranks = get_dst_ranks(2, 8, 3, 2);
  const std::vector<int32_t> expect_owner_ranks = {8};
  EXPECT_EQ(owner_ranks, expect_owner_ranks);

  const std::vector<int32_t> wrapped_owner_ranks = get_dst_ranks(5, 8, 3, 2);
  const std::vector<int32_t> expect_wrapped_owner_ranks = {8};
  EXPECT_EQ(wrapped_owner_ranks, expect_wrapped_owner_ranks);
}

TEST(PushRouteTest, HeteroTpTwoToOneKeepsOddDpRoute) {
  const std::vector<int32_t> odd_dp_ranks = get_dst_ranks(1, 2, 1, 3);
  const std::vector<int32_t> expect_odd_dp_ranks = {3};
  EXPECT_EQ(odd_dp_ranks, expect_odd_dp_ranks);
}

TEST(PushRouteTest, DstDpRankOffsetApplied) {
  const std::vector<int32_t> dst_ranks = get_dst_ranks(1, 6, 4, 3);
  const std::vector<int32_t> expect_ranks = {13};
  EXPECT_EQ(dst_ranks, expect_ranks);
}

TEST(PushRouteTest, DecodeTpOneLinksEveryPrefillTpRank) {
  const std::vector<int32_t> src_tp_ranks = get_src_tp_ranks(0, 2, 1);
  const std::vector<int32_t> expect_src_tp_ranks = {0, 1};
  EXPECT_EQ(src_tp_ranks, expect_src_tp_ranks);
}

TEST(PushRouteTest, DecodeRankLinksMatchingPrefillOwners) {
  const std::vector<int32_t> src_tp_ranks = get_src_tp_ranks(1, 8, 2);
  const std::vector<int32_t> expect_src_tp_ranks = {1, 3, 5, 7};
  EXPECT_EQ(src_tp_ranks, expect_src_tp_ranks);
}

TEST(PushRouteTest, LargerDecodeTpKeepsRoundRobinSource) {
  const std::vector<int32_t> src_tp_ranks = get_src_tp_ranks(5, 2, 8);
  const std::vector<int32_t> expect_src_tp_ranks = {1};
  EXPECT_EQ(src_tp_ranks, expect_src_tp_ranks);
}

}  // namespace xllm

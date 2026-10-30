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

#include "core/scheduler/dp_rank_selector.h"

#include <gtest/gtest.h>

#include <array>

namespace xllm {
namespace {

TEST(DpRankSelectorTest, PrefersAdmissibleHeadroomOverLogicalCapacity) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 3> loads = {{{100, 90}, {60, 10}, {200, 180}}};
  EXPECT_EQ(selector.select(loads), 1);
}

TEST(DpRankSelectorTest, RotatesOnlyAmongEqualBestCandidates) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 3> loads = {{{10, 8}, {20, 10}, {30, 20}}};
  EXPECT_EQ(selector.select(loads), 1);
  EXPECT_EQ(selector.select(loads), 2);
  EXPECT_EQ(selector.select(loads), 1);
}

TEST(DpRankSelectorTest, FiltersBlockedRanksBeforeScoring) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 3> loads = {{{100, 0}, {20, 1}, {20, 2}}};
  const std::array<uint8_t, 3> eligible = {0, 1, 1};
  EXPECT_EQ(selector.select(loads, {.eligible_ranks = eligible}), 1);
}

TEST(DpRankSelectorTest, PreservesAssignedRankWithLessHeadroom) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 2> loads = {{{10, 10}, {100, 0}}};
  EXPECT_EQ(selector.select(loads, {.assigned_rank = 0}), 0);
}

TEST(DpRankSelectorTest, BlockedBindingWaitsInsteadOfMigrating) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 2> loads = {{{10, 1}, {100, 0}}};
  const std::array<uint8_t, 2> eligible = {0, 1};
  EXPECT_EQ(
      selector.select(loads, {.assigned_rank = 0, .eligible_ranks = eligible}),
      -1);
}

TEST(DpRankSelectorTest, ReturnsNoRankWhenAllCandidatesAreBlocked) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 2> loads = {{{100, 0}, {100, 0}}};
  const std::array<uint8_t, 2> eligible = {0, 0};
  EXPECT_EQ(selector.select(loads, {.eligible_ranks = eligible}), -1);
  EXPECT_EQ(selector.select({}), -1);
}

TEST(DpRankSelectorTest, ClampsConcurrentUsageAboveCapacity) {
  DpRankSelector selector;
  const std::array<DpRankLoad, 2> loads = {{{10, 11}, {10, 10}}};
  EXPECT_EQ(selector.select(loads), 0);
  EXPECT_EQ(selector.select(loads), 1);
}

}  // namespace
}  // namespace xllm

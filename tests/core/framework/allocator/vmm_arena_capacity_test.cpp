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

#include "core/framework/allocator/vmm_arena_capacity.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace xllm {
namespace {

TEST(VmmArenaCapacityTest, ElasticCapacityIsIndependentOfUtilizationRatio) {
  for (double ratio : {0.75, 0.9, 1.0}) {
    SCOPED_TRACE(ratio);
    const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
        /*available_memory=*/20,
        /*total_memory=*/80,
        ratio,
        /*lazy=*/true,
        /*max_cache_size=*/0);

    EXPECT_EQ(capacity.maximum_bytes, 80);
    EXPECT_EQ(capacity.initial_bytes, 20);
    EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
  }
}

TEST(VmmArenaCapacityTest, ReduceHeterogeneousWorkerCapacitiesIndependently) {
  const VmmArenaCapacity first = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0);
  const VmmArenaCapacity second = calculate_vmm_arena_capacity(
      /*available_memory=*/36,
      /*total_memory=*/40,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0);

  EXPECT_EQ(second.maximum_bytes, 40);
  EXPECT_EQ(second.initial_bytes, 36);
  EXPECT_EQ(second.growth_memory_reserve_bytes, 0);
  EXPECT_EQ(std::min(first.maximum_bytes, second.maximum_bytes), 40);
  EXPECT_EQ(std::min(first.initial_bytes, second.initial_bytes), 20);
}

TEST(VmmArenaCapacityTest, RetainElasticArenaWithoutStartupFreeMemory) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/0,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.maximum_bytes, 80);
  EXPECT_EQ(capacity.initial_bytes, 0);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, ZeroRatioKeepsElasticCapacityUnchanged) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.0,
      /*lazy=*/true,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.maximum_bytes, 80);
  EXPECT_EQ(capacity.initial_bytes, 20);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, KeepPersistentStatesWithinStartupFreeMemory) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.75,
      /*lazy=*/true,
      /*max_cache_size=*/0,
      /*reserved_memory_bytes=*/0,
      /*resident_memory_bytes=*/8);

  EXPECT_EQ(capacity.maximum_bytes, 72);
  EXPECT_EQ(capacity.initial_bytes, 12);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, SeparateResidentStatesFromEncoderGrowthReserve) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/40,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0,
      /*reserved_memory_bytes=*/8,
      /*resident_memory_bytes=*/4);

  EXPECT_EQ(capacity.maximum_bytes, 68);
  EXPECT_EQ(capacity.initial_bytes, 28);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 8);
}

TEST(VmmArenaCapacityTest, ReserveResidentStatesOutsideFixedPool) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/40,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/false,
      /*max_cache_size=*/20,
      /*reserved_memory_bytes=*/0,
      /*resident_memory_bytes=*/8);

  EXPECT_EQ(capacity.maximum_bytes, 12);
  EXPECT_EQ(capacity.initial_bytes, 12);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, FullUtilizationKeepsFixedPoolWithinAvailableMemory) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/1.0,
      /*lazy=*/false,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.maximum_bytes, 20);
  EXPECT_EQ(capacity.initial_bytes, 20);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, ApplyCacheLimitWithoutIncreasingInitialSlots) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/32);

  EXPECT_EQ(capacity.maximum_bytes, 32);
  EXPECT_EQ(capacity.initial_bytes, 20);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, CacheLimitBoundsInitialSlotsBelowStartupFreeMemory) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/12);

  EXPECT_EQ(capacity.maximum_bytes, 12);
  EXPECT_EQ(capacity.initial_bytes, 12);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest,
     KeepExpandableArenaWhenExplicitReserveConsumesStartupFreeMemory) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/12,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0,
      /*reserved_memory_bytes=*/16);

  EXPECT_EQ(capacity.maximum_bytes, 64);
  EXPECT_EQ(capacity.initial_bytes, 0);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 16);
}

TEST(VmmArenaCapacityTest, PreserveFixedPoolRatioStartupBudget) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/false,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.maximum_bytes, 4);
  EXPECT_EQ(capacity.initial_bytes, 4);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, ClampExhaustedFixedPoolBudgetToZero) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/12,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/false,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.maximum_bytes, 0);
  EXPECT_EQ(capacity.initial_bytes, 0);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
}

TEST(VmmArenaCapacityTest, RoundStartupBufferUpBeforeRoundingPageCapacityDown) {
  constexpr int64_t kPageSize = 16;
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/41,
      /*total_memory=*/103,
      /*max_memory_utilization=*/0.75,
      /*lazy=*/false,
      /*max_cache_size=*/0);

  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 0);
  EXPECT_EQ(capacity.maximum_bytes, 15);
  EXPECT_EQ(capacity.initial_bytes, 15);
  EXPECT_EQ(capacity.maximum_bytes / kPageSize, 0);
  EXPECT_EQ(capacity.initial_bytes / kPageSize, 0);
}

TEST(VmmArenaCapacityTest, ReserveEncoderCacheOutsideFixedPool) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/40,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/false,
      /*max_cache_size=*/20,
      /*reserved_memory_bytes=*/8);

  EXPECT_EQ(capacity.maximum_bytes, 12);
  EXPECT_EQ(capacity.initial_bytes, 12);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 8);
}

TEST(VmmArenaCapacityTest, PreserveEncoderCacheDuringElasticGrowth) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/20,
      /*total_memory=*/80,
      /*max_memory_utilization=*/0.8,
      /*lazy=*/true,
      /*max_cache_size=*/0,
      /*reserved_memory_bytes=*/8);

  EXPECT_EQ(capacity.maximum_bytes, 72);
  EXPECT_EQ(capacity.initial_bytes, 12);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 8);
}

TEST(VmmArenaCapacityTest, EncoderReservationCanExhaustCacheLimit) {
  const VmmArenaCapacity capacity = calculate_vmm_arena_capacity(
      /*available_memory=*/40,
      /*total_memory=*/80,
      /*max_memory_utilization=*/1.0,
      /*lazy=*/true,
      /*max_cache_size=*/4,
      /*reserved_memory_bytes=*/8);

  EXPECT_EQ(capacity.maximum_bytes, 0);
  EXPECT_EQ(capacity.initial_bytes, 0);
  EXPECT_EQ(capacity.growth_memory_reserve_bytes, 8);
}

}  // namespace
}  // namespace xllm

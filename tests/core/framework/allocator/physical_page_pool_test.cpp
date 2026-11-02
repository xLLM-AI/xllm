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

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "core/framework/allocator/vmm_arena_capacity.h"

namespace xllm {

// The production growth boundary is exercised without initializing a device
// runtime. Stub pages own no driver handles, but use the real pool state
// machine.
class PhysicalPagePoolTestPeer final {
 public:
  static void inject(
      PhysicalPagePool& pool,
      std::function<int64_t()> memory_query,
      std::function<std::unique_ptr<PhysicalPage>(page_id_t)> factory) {
    std::lock_guard<std::mutex> lock(pool.mtx_);
    pool.free_memory_query_ = std::move(memory_query);
    pool.allocator_memory_query_ = []() {
      return std::optional<AllocatorMemoryStats>{};
    };
    pool.page_factory_ = std::move(factory);
  }

  static void inject_allocator_memory(
      PhysicalPagePool& pool,
      std::function<std::optional<AllocatorMemoryStats>()> memory_query) {
    std::lock_guard<std::mutex> lock(pool.mtx_);
    pool.allocator_memory_query_ = std::move(memory_query);
  }

  static std::unique_ptr<PhysicalPage> make_page(page_id_t page_id) {
    auto& pool = PhysicalPagePool::get_instance();
    return std::unique_ptr<PhysicalPage>(
        new PhysicalPage(pool.device_, pool.page_size_, page_id, nullptr));
  }
};

namespace {

constexpr size_t kCapacity = 6;
constexpr size_t kPageSize = 4096;

class PhysicalPagePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& pool = PhysicalPagePool::get_instance();
    ASSERT_TRUE(pool.reset());
    pool.init(torch::Device(torch::kCPU), kCapacity, kPageSize, /*lazy=*/true);
    inject_factory();
  }

  void TearDown() override {
    EXPECT_TRUE(PhysicalPagePool::get_instance().reset());
  }

  void inject_factory() {
    PhysicalPagePoolTestPeer::inject(
        PhysicalPagePool::get_instance(),
        [this]() { return free_bytes_; },
        [this](page_id_t page_id) {
          ++allocation_attempts_;
          if (fail_on_attempt_ == allocation_attempts_) {
            return std::unique_ptr<PhysicalPage>{};
          }
          return PhysicalPagePoolTestPeer::make_page(page_id);
        });
  }

  void reset_with_active_pages(size_t initial_pages) {
    auto& pool = PhysicalPagePool::get_instance();
    ASSERT_TRUE(pool.reset());
    pool.init(torch::Device(torch::kCPU),
              kCapacity,
              initial_pages,
              kPageSize,
              /*lazy=*/true);
    inject_factory();
  }

  void inject_allocator_stats() {
    PhysicalPagePoolTestPeer::inject_allocator_memory(
        PhysicalPagePool::get_instance(),
        [this]() { return allocator_stats_; });
  }

  int64_t free_bytes_ = kCapacity * kPageSize;
  size_t allocation_attempts_ = 0;
  size_t fail_on_attempt_ = 0;
  std::optional<AllocatorMemoryStats> allocator_stats_;
};

TEST_F(PhysicalPagePoolTest, LazyInitializationSeparatesCapacityAndBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_total(), kCapacity);
  EXPECT_EQ(pool.num_active(), kCapacity);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_allocatable(), kCapacity);
  EXPECT_EQ(allocation_attempts_, 0);
  for (const PhysicalPage* page : pool.get_all_pages()) {
    EXPECT_EQ(page, nullptr);
  }
  EXPECT_TRUE(pool.batch_get(/*count=*/0).empty());
  EXPECT_TRUE(pool.batch_get(kCapacity + 1).empty());
  EXPECT_EQ(allocation_attempts_, 0);
}

TEST_F(PhysicalPagePoolTest, ActivatesAdditionalSlotsWhenStartupArenaIsFull) {
  reset_with_active_pages(/*initial_pages=*/2);
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_total(), kCapacity);
  EXPECT_EQ(pool.num_active(), 2);
  EXPECT_EQ(pool.num_available(), 2);
  EXPECT_EQ(pool.num_allocatable(), kCapacity);
  EXPECT_EQ(pool.get_all_pages().size(), kCapacity);

  auto startup_pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(startup_pages.size(), 2);
  const PhysicalPage* first_page = startup_pages.front().get();
  EXPECT_EQ(pool.num_available(), 0);
  auto growth_pages = pool.batch_get(/*count=*/3);
  ASSERT_EQ(growth_pages.size(), 3);
  EXPECT_EQ(growth_pages.front()->page_id(), 2);
  EXPECT_EQ(growth_pages.back()->page_id(), 4);
  EXPECT_EQ(pool.num_active(), 5);
  EXPECT_EQ(pool.num_committed(), 5);
  EXPECT_EQ(startup_pages.front().get(), first_page);
  EXPECT_FALSE(pool.reset());

  pool.batch_put(startup_pages);
  pool.batch_put(growth_pages);
  EXPECT_EQ(pool.num_available(), 5);
  EXPECT_EQ(pool.trim_idle(), 5);
  EXPECT_EQ(pool.num_active(), 5);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_allocatable(), kCapacity);
}

TEST_F(PhysicalPagePoolTest, CanGrowFromAnEmptyStartupArena) {
  reset_with_active_pages(/*initial_pages=*/0);
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_available(), 0);
  EXPECT_EQ(pool.num_active(), 0);
  auto page = pool.get();
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(page->page_id(), 0);
  EXPECT_EQ(pool.num_active(), 1);
  pool.put(std::move(page));
  EXPECT_EQ(pool.num_available(), 1);
}

TEST_F(PhysicalPagePoolTest, ArenaGrowthPreservesLifetimeWorkspaceRecovery) {
  reset_with_active_pages(/*initial_pages=*/2);
  auto& pool = PhysicalPagePool::get_instance();
  free_bytes_ = 4 * kPageSize;
  pool.observe_execution_memory();
  // Recovered workspace remains reserved even when inactive slots exist.
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 4);
  EXPECT_TRUE(pool.batch_get(/*count=*/5).empty());
  EXPECT_EQ(pool.num_active(), 2);
  EXPECT_EQ(allocation_attempts_, 0);
  auto pages = pool.batch_get(/*count=*/3);
  ASSERT_EQ(pages.size(), 3);
  EXPECT_EQ(pool.num_active(), 3);
  free_bytes_ = 3 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 1);
  EXPECT_TRUE(pool.batch_get(/*count=*/2).empty());
  EXPECT_EQ(pool.num_active(), 3);
  pool.batch_put(pages);
  // Further device-memory recovery must not weaken the lifetime watermark.
  free_bytes_ = 5 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 4);
  EXPECT_TRUE(pool.batch_get(/*count=*/5).empty());
}

TEST_F(PhysicalPagePoolTest, LargerAllocatorPeakPreventsUnsafeSlotActivation) {
  reset_with_active_pages(/*initial_pages=*/1);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{2 * kPageSize, 2 * kPageSize};
  free_bytes_ = 4 * kPageSize;
  pool.observe_execution_memory();
  allocator_stats_ = AllocatorMemoryStats{0, 2 * kPageSize};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 4);
  // Memory freed outside the allocator permits a larger execution workspace.
  allocator_stats_ = AllocatorMemoryStats{4 * kPageSize, 4 * kPageSize};
  free_bytes_ = 6 * kPageSize;
  pool.observe_execution_memory();
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  EXPECT_EQ(pool.num_allocatable(), 2);
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(allocation_attempts_, 0);
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  EXPECT_EQ(pool.num_active(), 2);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, OutsideMemoryReleaseAllowsAdditionalArenaGrowth) {
  reset_with_active_pages(/*initial_pages=*/2);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{2 * kPageSize, 2 * kPageSize};
  free_bytes_ = 4 * kPageSize;
  pool.observe_execution_memory();
  allocator_stats_ = AllocatorMemoryStats{0, 2 * kPageSize};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 4);
  // Driver recovery outside the allocator must not reserve arbitrary bytes.
  free_bytes_ = 10 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), kCapacity);
  auto pages = pool.batch_get(/*count=*/5);
  ASSERT_EQ(pages.size(), 5);
  EXPECT_EQ(pool.num_active(), 5);
  EXPECT_EQ(pages.back()->page_id(), 4);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, AllocatorCacheAccumulationAvoidsDoubleCharging) {
  reset_with_active_pages(/*initial_pages=*/1);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{2 * kPageSize, 2 * kPageSize};
  free_bytes_ = 4 * kPageSize;
  pool.observe_execution_memory();
  allocator_stats_ = AllocatorMemoryStats{4 * kPageSize, 4 * kPageSize};
  free_bytes_ = 2 * kPageSize;
  pool.observe_execution_memory();
  EXPECT_EQ(pool.num_allocatable(), 2);
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  free_bytes_ = 0;
  pool.batch_put(pages);
  EXPECT_EQ(pool.num_allocatable(), 2);
  // Released cache adds free memory but requires the same recovery reserve.
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest,
       GrowthRechecksAllocatorPeakAfterAdmissionSnapshot) {
  reset_with_active_pages(/*initial_pages=*/1);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{0, 2 * kPageSize};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 4);
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(allocation_attempts_, 0);
  // A backend reset between admission and growth cannot weaken the reserve.
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  EXPECT_EQ(pool.num_allocatable(), 2);
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
}

TEST_F(PhysicalPagePoolTest, ObserveQueriesEachMemorySourceOnce) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  size_t driver_queries = 0;
  size_t allocator_queries = 0;
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this, &driver_queries]() {
        ++driver_queries;
        return free_bytes_;
      },
      [](page_id_t page_id) {
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  PhysicalPagePoolTestPeer::inject_allocator_memory(
      pool, [this, &allocator_queries]() {
        ++allocator_queries;
        return allocator_stats_;
      });
  allocator_stats_ = AllocatorMemoryStats{4 * kPageSize, 4 * kPageSize};
  free_bytes_ = -1;
  pool.observe_execution_memory();
  EXPECT_EQ(driver_queries, 1);
  EXPECT_EQ(allocator_queries, 1);
  // Allocator peaks remain useful even when the driver query fails.
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
  EXPECT_EQ(driver_queries, 2);
  EXPECT_EQ(allocator_queries, 2);
}

TEST_F(PhysicalPagePoolTest, AdmissionRetainsPeakWithNoDriverMemoryAvailable) {
  reset_with_active_pages(/*initial_pages=*/1);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 0);
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest, FailedDriverGrowthQueryStillRetainsAllocatorPeak) {
  reset_with_active_pages(/*initial_pages=*/1);
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  free_bytes_ = -1;
  EXPECT_EQ(pool.get(), nullptr);
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(allocation_attempts_, 0);
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest, FullArenaAdmissionRetainsPeakBeforeTrimming) {
  inject_allocator_stats();
  auto& pool = PhysicalPagePool::get_instance();
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  auto pages = pool.batch_get(kCapacity);
  ASSERT_EQ(pages.size(), kCapacity);
  allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
  EXPECT_EQ(pool.num_allocatable(), 0);
  pool.batch_put(pages);
  EXPECT_EQ(pool.trim_idle(), kCapacity);
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest, PostAllocationBudgetRejectsRacingAllocatorGrowth) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  size_t driver_queries = 0;
  allocator_stats_ = AllocatorMemoryStats{2 * kPageSize, 4 * kPageSize};
  free_bytes_ = 6 * kPageSize;
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this, &driver_queries]() {
        const int64_t result = free_bytes_;
        if (++driver_queries == 1) {
          // Another allocator consumes memory between the two prechecks.
          allocator_stats_ = AllocatorMemoryStats{3 * kPageSize, 4 * kPageSize};
          free_bytes_ = 5 * kPageSize;
        }
        return result;
      },
      [this](page_id_t page_id) {
        ++allocation_attempts_;
        free_bytes_ -= kPageSize;
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  inject_allocator_stats();
  EXPECT_TRUE(pool.batch_get(/*count=*/5).empty());
  EXPECT_EQ(allocation_attempts_, 5);
  EXPECT_EQ(driver_queries, 2);
  EXPECT_EQ(free_bytes_, 0);
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
  for (const PhysicalPage* page : pool.get_all_pages()) {
    EXPECT_EQ(page, nullptr);
  }
  // Stub handles do not return driver memory on destruction. The unpublished
  // ownership and activation checks above verify rollback independently.
}

TEST_F(PhysicalPagePoolTest, PostAllocationDriverFailurePublishesNoBacking) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this]() { return free_bytes_; },
      [this](page_id_t page_id) {
        ++allocation_attempts_;
        free_bytes_ = -1;
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  EXPECT_TRUE(pool.batch_get(/*count=*/2).empty());
  EXPECT_EQ(allocation_attempts_, 2);
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
}

TEST_F(PhysicalPagePoolTest, PostAllocationUsesConservativeAllocatorSnapshots) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  size_t driver_queries = 0;
  allocator_stats_ = AllocatorMemoryStats{0, 0};
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this, &driver_queries]() {
        const int64_t result = free_bytes_;
        if (++driver_queries == 2) {
          allocator_stats_ = AllocatorMemoryStats{4 * kPageSize, 4 * kPageSize};
        }
        return result;
      },
      [this](page_id_t page_id) {
        ++allocation_attempts_;
        free_bytes_ = 3 * kPageSize;
        allocator_stats_ = AllocatorMemoryStats{0, 4 * kPageSize};
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  inject_allocator_stats();
  EXPECT_EQ(pool.get(), nullptr);
  EXPECT_EQ(allocation_attempts_, 1);
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
}

TEST_F(PhysicalPagePoolTest, PostAllocationPreservesFallbackRecoveryBudget) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  free_bytes_ = 4 * kPageSize;
  pool.observe_execution_memory();
  free_bytes_ = 6 * kPageSize;
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this]() { return free_bytes_; },
      [this](page_id_t page_id) {
        ++allocation_attempts_;
        free_bytes_ -= kPageSize;
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  auto pages = pool.batch_get(/*count=*/4);
  ASSERT_EQ(pages.size(), 4);
  EXPECT_EQ(free_bytes_, 2 * kPageSize);
  EXPECT_EQ(pool.num_active(), 4);
  EXPECT_EQ(pool.num_committed(), 4);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest,
       FailedArenaGrowthLeavesActivationAndOwnershipIntact) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  auto page = pool.get();
  ASSERT_NE(page, nullptr);
  const PhysicalPage* original = page.get();
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_available(), 0);
  EXPECT_EQ(pool.num_committed(), 1);
  EXPECT_EQ(pool.get_all_pages()[0], original);
  EXPECT_EQ(pool.get_all_pages()[1], nullptr);
  pool.put(std::move(page));
  fail_on_attempt_ = 0;
  auto pages = pool.batch_get(/*count=*/3);
  ASSERT_EQ(pages.size(), 3);
  EXPECT_EQ(pages.front().get(), original);
  EXPECT_EQ(pool.num_active(), 3);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, RejectsGrowthBeyondTheReservedArena) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  auto pages = pool.batch_get(kCapacity);
  ASSERT_EQ(pages.size(), kCapacity);
  EXPECT_EQ(pool.num_active(), kCapacity);
  EXPECT_EQ(pool.num_allocatable(), 0);
  EXPECT_EQ(pool.get(), nullptr);
  EXPECT_EQ(pool.allocate_contiguous_from_right(/*count=*/1), -1);
  EXPECT_TRUE(pool.allocate_pages_from_right(/*count=*/1).empty());
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, WeightGrowthUsesStableIdsAtTheRightArenaBoundary) {
  reset_with_active_pages(/*initial_pages=*/2);
  auto& pool = PhysicalPagePool::get_instance();
  auto page = pool.get();
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(page->page_id(), 0);
  const page_id_t start = pool.allocate_contiguous_from_right(/*count=*/2);
  ASSERT_EQ(start, 4);
  EXPECT_EQ(pool.num_active(), 4);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_committed(), 3);
  const auto ids = pool.allocate_pages_from_right(/*count=*/2);
  ASSERT_EQ(ids, std::vector<page_id_t>({3, 2}));
  EXPECT_EQ(pool.num_active(), kCapacity);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_allocatable(), 1);
  pool.release_reserved_pages(ids);
  pool.release_reserved_pages({start, start + 1});
  pool.put(std::move(page));
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.trim_idle(), 5);
}

TEST_F(PhysicalPagePoolTest, FailedWeightGrowthDoesNotActivateRightHandSlots) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  fail_on_attempt_ = 2;
  EXPECT_EQ(pool.allocate_contiguous_from_right(/*count=*/3), -1);
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.allocate_pages_from_right(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(pool.num_available(), 1);
  EXPECT_EQ(pool.num_committed(), 0);
}

TEST_F(PhysicalPagePoolTest,
       GrowthReserveAndExecutionHeadroomUseTheLargerLimit) {
  reset_with_active_pages(/*initial_pages=*/1);
  auto& pool = PhysicalPagePool::get_instance();
  pool.set_growth_reserve(2 * kPageSize);
  pool.set_allocation_headroom(kPageSize);
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 2);
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_active(), 1);
  EXPECT_EQ(allocation_attempts_, 0);
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 2);
  pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  free_bytes_ = 4 * kPageSize;
  pool.set_allocation_headroom(3 * kPageSize);
  EXPECT_EQ(pool.num_allocatable(), 3);
  EXPECT_TRUE(pool.batch_get(/*count=*/4).empty());
  EXPECT_EQ(pool.num_active(), 2);
  EXPECT_EQ(pool.num_committed(), 2);
}

TEST_F(PhysicalPagePoolTest, MapsPaddingPagesAfterPersistentStateAllocation) {
  constexpr int64_t kAvailableBytes = 100 * kPageSize;
  constexpr int64_t kResidentStateBytes = 60 * kPageSize;
  constexpr size_t kPaddingPages = 34;
  const VmmArenaCapacity capacity =
      calculate_vmm_arena_capacity(kAvailableBytes,
                                   /*total_memory=*/200 * kPageSize,
                                   /*max_memory_utilization=*/0.0,
                                   /*max_memory_utilization_configured=*/true,
                                   /*lazy=*/true,
                                   /*max_cache_size=*/0,
                                   /*reserved_memory_bytes=*/0,
                                   kResidentStateBytes);
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_TRUE(pool.reset());
  pool.init(torch::Device(torch::kCPU),
            capacity.maximum_bytes / kPageSize,
            capacity.initial_bytes / kPageSize,
            kPageSize,
            /*lazy=*/true);
  pool.set_growth_reserve(capacity.growth_memory_reserve_bytes);
  // CONV/SSM tensors become resident before the first padding page group.
  free_bytes_ = kAvailableBytes - kResidentStateBytes;
  PhysicalPagePoolTestPeer::inject(
      pool,
      [this]() { return free_bytes_; },
      [this](page_id_t page_id) {
        free_bytes_ -= kPageSize;
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });

  auto pages = pool.batch_get(kPaddingPages);
  ASSERT_EQ(pages.size(), kPaddingPages);
  EXPECT_EQ(pool.num_committed(), kPaddingPages);
  EXPECT_EQ(free_bytes_, 6 * kPageSize);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, PartialAllocationFailurePublishesNoBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  fail_on_attempt_ = 2;
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(allocation_attempts_, 2);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 0);
  for (const PhysicalPage* page : pool.get_all_pages()) {
    EXPECT_EQ(page, nullptr);
  }
  fail_on_attempt_ = 0;
  auto pages = pool.batch_get(/*count=*/3);
  ASSERT_EQ(pages.size(), 3);
  EXPECT_EQ(pages.front()->page_id(), 0);
  EXPECT_EQ(pool.num_committed(), 3);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, FailedGrowthPreservesResidentBackingAndFreeSlots) {
  auto& pool = PhysicalPagePool::get_instance();
  auto page = pool.get();
  ASSERT_NE(page, nullptr);
  const PhysicalPage* original = page.get();
  pool.put(std::move(page));
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 1);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.get_all_pages()[0], original);
  EXPECT_EQ(pool.get_all_pages()[1], nullptr);
}

TEST_F(PhysicalPagePoolTest,
       HeadroomChecksGrowthWithoutChargingIdlePagesAgain) {
  auto& pool = PhysicalPagePool::get_instance();
  pool.set_allocation_headroom(kPageSize);
  free_bytes_ = 3 * kPageSize - 1;
  EXPECT_EQ(pool.num_allocatable(), 1);
  EXPECT_TRUE(pool.batch_get(/*count=*/2).empty());
  EXPECT_EQ(allocation_attempts_, 0);
  free_bytes_ = 3 * kPageSize;
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 2);
  pages = pool.batch_get(/*count=*/2);
  EXPECT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  pool.set_allocation_headroom(std::numeric_limits<size_t>::max());
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 2);
}

TEST_F(PhysicalPagePoolTest,
       ExecutionWatermarkAvoidsChargingResidentCachesTwice) {
  auto& pool = PhysicalPagePool::get_instance();
  free_bytes_ = 3 * kPageSize;
  pool.observe_execution_memory();
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 3);
  EXPECT_TRUE(pool.batch_get(/*count=*/4).empty());
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  // KV growth reduces free bytes but does not increase execution reserve.
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 1);
  // Resident workspace already reduces free bytes; do not charge it twice.
  free_bytes_ = kPageSize;
  pool.observe_execution_memory();
  EXPECT_EQ(pool.num_allocatable(), 1);
  pool.batch_put(pages);
  EXPECT_EQ(pool.num_allocatable(), 3);
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 3);
  pool.set_allocation_headroom(kPageSize);
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest, TrimmingKeepsLiveIdsAndConfiguredIdleBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  auto pages = pool.batch_get(/*count=*/4);
  ASSERT_EQ(pages.size(), 4);
  const page_id_t active_id = pages.front()->page_id();
  const PhysicalPage* active_ptr = pages.front().get();
  for (size_t i = 1; i < pages.size(); ++i) {
    pool.put(std::move(pages[i]));
  }
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/1), 2);
  EXPECT_EQ(pool.num_total(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pages.front()->page_id(), active_id);
  EXPECT_EQ(pages.front().get(), active_ptr);
  EXPECT_FALSE(pool.reset());
  pool.batch_put(pages);
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 2);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
}

TEST_F(PhysicalPagePoolTest, WeightReservationsPinBackingUntilReleased) {
  auto& pool = PhysicalPagePool::get_instance();
  const page_id_t start = pool.allocate_contiguous_from_right(/*count=*/2);
  ASSERT_EQ(start, static_cast<page_id_t>(kCapacity - 2));
  auto kv_page = pool.get();
  ASSERT_NE(kv_page, nullptr);
  pool.put(std::move(kv_page));
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 1);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pool.num_available(), kCapacity - 2);
  pool.release_reserved_pages({start, start + 1});
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 2);
  // Returned resident weight slots must be reused before empty logical slots.
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  EXPECT_GE(pages.front()->page_id(), start);
  pool.batch_put(pages);
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 2);
}

TEST_F(PhysicalPagePoolTest, FailedWeightReservationLeavesNoPinnedSlots) {
  auto& pool = PhysicalPagePool::get_instance();
  fail_on_attempt_ = 2;
  EXPECT_EQ(pool.allocate_contiguous_from_right(/*count=*/3), -1);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.allocate_pages_from_right(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
}

TEST_F(PhysicalPagePoolTest, ConcurrentGrowthRechecksMemoryUnderOnePoolLock) {
  auto& pool = PhysicalPagePool::get_instance();
  std::atomic<int64_t> free_bytes(3 * kPageSize);
  PhysicalPagePoolTestPeer::inject(
      pool,
      [&free_bytes]() { return free_bytes.load(); },
      [&free_bytes](page_id_t page_id) {
        free_bytes.fetch_sub(kPageSize);
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  std::vector<std::unique_ptr<PhysicalPage>> first;
  std::vector<std::unique_ptr<PhysicalPage>> second;
  std::thread first_thread([&]() { first = pool.batch_get(/*count=*/2); });
  std::thread second_thread([&]() { second = pool.batch_get(/*count=*/2); });
  first_thread.join();
  second_thread.join();
  EXPECT_EQ(first.size() + second.size(), 2);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pool.num_available(), kCapacity - 2);
  pool.batch_put(first);
  pool.batch_put(second);
}

}  // namespace
}  // namespace xllm

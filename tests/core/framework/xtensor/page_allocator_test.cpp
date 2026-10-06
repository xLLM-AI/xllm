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

#include "core/framework/xtensor/page_allocator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace xllm {

// Seed only completed mapping states. Tests exercise production allocation,
// release, and worker accounting without requiring an RPC service or device.
class PageAllocatorTestPeer final {
 public:
  using AllocatorPtr = std::unique_ptr<PageAllocator, void (*)(PageAllocator*)>;

  static AllocatorPtr create(size_t num_phy_pages,
                             int32_t dp_size,
                             int32_t world_size) {
    AllocatorPtr allocator(new PageAllocator(),
                           [](PageAllocator* value) { delete value; });
    allocator->init(num_phy_pages,
                    dp_size,
                    world_size,
                    /*enable_page_prealloc=*/false);
    return allocator;
  }

  static void reserve_pages(PageAllocator& allocator,
                            const std::string& model_id,
                            int32_t dp_rank,
                            size_t num_pages) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    auto& pages = state.dp_group_pages[dp_rank];
    ASSERT_GE(pages.free_virt_page_list.size(), num_pages);
    ASSERT_TRUE(allocator.consume_phy_pages_for_dp(
        model_id, dp_rank, num_pages * state.phy_pages_per_virt_page));
    for (size_t index = 0; index < num_pages; ++index) {
      pages.reserved_virt_page_list.emplace_back(
          pages.free_virt_page_list.front());
      pages.free_virt_page_list.pop_front();
    }
  }

  static void finish_sleep(PageAllocator& allocator,
                           const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    state.is_sleeping = true;
    state.kv_cache_mapped = false;
    for (int32_t dp_rank = 0; dp_rank < allocator.dp_size_; ++dp_rank) {
      const auto& pages = state.dp_group_pages[dp_rank];
      size_t mapped_pages = pages.reserved_virt_page_list.size() +
                            pages.allocated_virt_page_list.size();
      allocator.release_phy_pages_for_dp(
          model_id, dp_rank, mapped_pages * state.phy_pages_per_virt_page);
    }
    state.transition = PageAllocator::ModelTransition::IDLE;
    allocator.cond_.notify_all();
  }

  static void begin_pending_map(PageAllocator& allocator,
                                const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    allocator.get_model_state(model_id).pending_map_ops.fetch_add(1);
  }

  static void finish_pending_map(PageAllocator& allocator,
                                 const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    allocator.get_model_state(model_id).pending_map_ops.fetch_sub(1);
    allocator.cond_.notify_all();
  }

  static bool wait_until_sleep_started(PageAllocator& allocator,
                                       const std::string& model_id) {
    std::unique_lock<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    return allocator.cond_.wait_for(lock, std::chrono::seconds(2), [&state] {
      return state.transition == PageAllocator::ModelTransition::SLEEPING;
    });
  }

  static void begin_wakeup(PageAllocator& allocator,
                           const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    state.is_sleeping = true;
    state.kv_cache_mapped = false;
    state.transition = PageAllocator::ModelTransition::WAKING;
  }

  static void finish_wakeup(PageAllocator& allocator,
                            const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    state.is_sleeping = false;
    state.kv_cache_mapped = true;
    state.transition = PageAllocator::ModelTransition::IDLE;
    allocator.cond_.notify_all();
  }

  static void begin_sleep_unmap(PageAllocator& allocator,
                                const std::string& model_id) {
    std::lock_guard<std::mutex> lock(allocator.mtx_);
    auto& state = allocator.get_model_state(model_id);
    state.is_sleeping = true;
    state.transition = PageAllocator::ModelTransition::SLEEPING;
  }
};

namespace {

TEST(PageAllocatorTest, VirtualPageExhaustionReturnsNullWithoutChangingCounts) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "model", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "model", /*dp_rank=*/0, /*num_pages=*/2);

  auto first = allocator->alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto second = allocator->alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->page_id(), 0);
  EXPECT_EQ(second->page_id(), 1);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 0);

  EXPECT_NO_THROW(EXPECT_EQ(
      allocator->alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr));
  EXPECT_EQ(allocator->get_num_free_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("model", /*dp_rank=*/0), 2);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 0);

  allocator->free_kv_cache_pages("model", /*dp_rank=*/0, {first->page_id()});
  auto reused = allocator->alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->page_id(), first->page_id());
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 0);
}

TEST(PageAllocatorTest, PhysicalPageExhaustionPreservesFreeVirtualPages) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "first", /*num_layers=*/2, MasterStatus::WAKEUP));
  ASSERT_TRUE(allocator->register_model(
      "second", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "first", /*dp_rank=*/0, /*num_pages=*/1);
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "second", /*dp_rank=*/0, /*num_pages=*/1);

  auto page = allocator->alloc_kv_cache_page("first", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  EXPECT_NO_THROW(EXPECT_EQ(
      allocator->alloc_kv_cache_page("first", /*dp_rank=*/0), nullptr));
  EXPECT_EQ(allocator->get_num_free_virt_pages("first", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("first", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator->get_num_reserved_virt_pages("second", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator->get_all_worker_free_pages(), std::vector<size_t>({0}));
}

TEST(PageAllocatorTest, DpGroupsConsumeOnlyTheirTpWorkers) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/2, /*world_size=*/4);
  ASSERT_TRUE(allocator->register_model(
      "model", /*num_layers=*/2, MasterStatus::WAKEUP));
  allocator->set_model_parallel_strategy("model", /*dp_size=*/2, /*tp_size=*/2);
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "model", /*dp_rank=*/0, /*num_pages=*/2);
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "model", /*dp_rank=*/1, /*num_pages=*/1);

  EXPECT_EQ(allocator->get_all_worker_free_pages(),
            std::vector<size_t>({0, 0, 4, 4}));
  auto page = allocator->alloc_kv_cache_page("model", /*dp_rank=*/1);
  ASSERT_NE(page, nullptr);
  allocator->free_kv_cache_pages("model", /*dp_rank=*/1, {page->page_id()});
  EXPECT_EQ(allocator->get_all_worker_free_pages(),
            std::vector<size_t>({0, 0, 4, 4}));
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("model", /*dp_rank=*/1), 0);
}

TEST(PageAllocatorTest, FreeSleepingPageDoesNotReleaseAnotherModelsPages) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "sleeping", /*num_layers=*/2, MasterStatus::WAKEUP));
  ASSERT_TRUE(allocator->register_model(
      "awake", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "sleeping", /*dp_rank=*/0, /*num_pages=*/2);
  auto page = allocator->alloc_kv_cache_page("sleeping", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);

  PageAllocatorTestPeer::finish_sleep(*allocator, "sleeping");
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "awake", /*dp_rank=*/0, /*num_pages=*/1);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 4);

  allocator->free_kv_cache_pages("sleeping", /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 4);
  EXPECT_EQ(allocator->get_num_free_virt_pages("sleeping", /*dp_rank=*/0), 2);
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("sleeping", /*dp_rank=*/0), 0);

  allocator->trim_kv_cache("sleeping", /*dp_rank=*/0);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 4);
  EXPECT_EQ(allocator->get_num_reserved_virt_pages("sleeping", /*dp_rank=*/0),
            0);
}

TEST(PageAllocatorTest, WakeupWaitsForSleepToDrainPendingMappings) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "model", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::begin_pending_map(*allocator, "model");

  auto sleep = std::async(std::launch::async, [&allocator] {
    return allocator->sleep_model("model", /*skip_weight_release=*/true);
  });
  EXPECT_TRUE(
      PageAllocatorTestPeer::wait_until_sleep_started(*allocator, "model"));

  std::promise<void> wakeup_started;
  auto wakeup = std::async(std::launch::async, [&allocator, &wakeup_started] {
    wakeup_started.set_value();
    return allocator->wakeup_model("model");
  });
  wakeup_started.get_future().wait();
  EXPECT_EQ(wakeup.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);

  PageAllocatorTestPeer::finish_pending_map(*allocator, "model");
  EXPECT_TRUE(sleep.get());
  EXPECT_TRUE(wakeup.get());
  EXPECT_FALSE(allocator->is_model_sleeping("model"));
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 8);
}

TEST(PageAllocatorTest, FreeWaitsForWakeupBeforeRecyclingMappedPage) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "model", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "model", /*dp_rank=*/0, /*num_pages=*/1);
  auto page = allocator->alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  // Wakeup has reserved its physical pages and is mapping this allocated ID.
  PageAllocatorTestPeer::begin_wakeup(*allocator, "model");

  std::promise<void> free_started;
  auto freeing =
      std::async(std::launch::async, [&allocator, &page, &free_started] {
        free_started.set_value();
        allocator->free_kv_cache_pages(
            "model", /*dp_rank=*/0, {page->page_id()});
      });
  free_started.get_future().wait();
  EXPECT_EQ(freeing.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("model", /*dp_rank=*/0), 1);

  PageAllocatorTestPeer::finish_wakeup(*allocator, "model");
  freeing.get();
  EXPECT_EQ(allocator->get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator->get_num_reserved_virt_pages("model", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 4);
  auto reused = allocator->alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->page_id(), page->page_id());
}

TEST(PageAllocatorTest, TrimWaitsForSleepBeforeReleasingVirtualIds) {
  auto allocator = PageAllocatorTestPeer::create(
      /*num_phy_pages=*/8, /*dp_size=*/1, /*world_size=*/1);
  ASSERT_TRUE(allocator->register_model(
      "model", /*num_layers=*/2, MasterStatus::WAKEUP));
  PageAllocatorTestPeer::reserve_pages(
      *allocator, "model", /*dp_rank=*/0, /*num_pages=*/1);
  PageAllocatorTestPeer::begin_sleep_unmap(*allocator, "model");

  std::promise<void> trim_started;
  auto trimming = std::async(std::launch::async, [&allocator, &trim_started] {
    trim_started.set_value();
    allocator->trim_kv_cache("model", /*dp_rank=*/0);
  });
  trim_started.get_future().wait();
  EXPECT_EQ(trimming.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(allocator->get_num_reserved_virt_pages("model", /*dp_rank=*/0), 1);

  PageAllocatorTestPeer::finish_sleep(*allocator, "model");
  trimming.get();
  EXPECT_EQ(allocator->get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator->get_num_free_phy_pages(), 8);
  EXPECT_EQ(allocator->get_num_free_virt_pages("model", /*dp_rank=*/0), 2);
}

}  // namespace
}  // namespace xllm

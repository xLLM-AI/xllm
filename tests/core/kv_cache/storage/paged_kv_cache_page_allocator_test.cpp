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

#include "core/kv_cache/storage/paged_kv_cache_page_allocator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xllm {
namespace {

class FakeKVCachePageBackend final : public KVCachePageBackend {
 public:
  FakeKVCachePageBackend(size_t total_pages, int32_t dp_size)
      : total_pages_(total_pages), free_pages_(dp_size, total_pages) {}

  bool try_reserve_pages(const std::string& /*model_id*/,
                         int32_t dp_rank,
                         size_t physical_pages) override {
    std::lock_guard<std::mutex> lock(mtx_);
    if (free_pages_[dp_rank] < physical_pages) {
      return false;
    }
    free_pages_[dp_rank] -= physical_pages;
    return true;
  }

  void release_pages(const std::string& /*model_id*/,
                     int32_t dp_rank,
                     size_t physical_pages) override {
    std::lock_guard<std::mutex> lock(mtx_);
    EXPECT_LE(physical_pages, total_pages_ - free_pages_[dp_rank]);
    free_pages_[dp_rank] += physical_pages;
  }

  size_t available_pages(const std::string& /*model_id*/,
                         int32_t dp_rank) const override {
    std::lock_guard<std::mutex> lock(mtx_);
    return free_pages_[dp_rank];
  }

  bool map_pages(const std::string& /*model_id*/,
                 int32_t dp_rank,
                 const std::vector<int64_t>& byte_offsets) override {
    std::unique_lock<std::mutex> lock(mtx_);
    map_dp_ranks_.emplace_back(dp_rank);
    mapped_offsets_.insert(
        mapped_offsets_.end(), byte_offsets.begin(), byte_offsets.end());
    map_started_ = true;
    cond_.notify_all();
    cond_.wait(lock, [this] { return !block_maps_; });
    return true;
  }

  bool unmap_pages(const std::string& /*model_id*/,
                   int32_t /*dp_rank*/,
                   const std::vector<int64_t>& byte_offsets) override {
    std::lock_guard<std::mutex> lock(mtx_);
    unmapped_offsets_.insert(
        unmapped_offsets_.end(), byte_offsets.begin(), byte_offsets.end());
    return true;
  }

  void block_maps() {
    std::lock_guard<std::mutex> lock(mtx_);
    block_maps_ = true;
    map_started_ = false;
  }

  bool wait_until_map_started() {
    std::unique_lock<std::mutex> lock(mtx_);
    return cond_.wait_for(
        lock, std::chrono::seconds(2), [this] { return map_started_; });
  }

  void release_maps() {
    std::lock_guard<std::mutex> lock(mtx_);
    block_maps_ = false;
    cond_.notify_all();
  }

  std::vector<int64_t> mapped_offsets() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return mapped_offsets_;
  }

  std::vector<int32_t> map_dp_ranks() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return map_dp_ranks_;
  }

 private:
  size_t total_pages_;
  std::vector<size_t> free_pages_;
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  bool block_maps_ = false;
  bool map_started_ = false;
  std::vector<int64_t> mapped_offsets_;
  std::vector<int64_t> unmapped_offsets_;
  std::vector<int32_t> map_dp_ranks_;
};

// Emulate the residency coordinator's completed lifecycle budget transaction.
void complete_suspend(PagedKVCachePageAllocator& allocator,
                      FakeKVCachePageBackend& backend,
                      const std::string& model_id,
                      const KVCacheMappingPlan& plan) {
  allocator.unmap_pages(model_id, plan);
  for (size_t dp_rank = 0; dp_rank < plan.dp_group_page_ids.size(); ++dp_rank) {
    backend.release_pages(model_id,
                          static_cast<int32_t>(dp_rank),
                          plan.dp_group_page_ids[dp_rank].size() *
                              plan.physical_pages_per_virtual_page);
  }
  allocator.finish_suspend(model_id);
}

bool wait_until_suspended(const PagedKVCachePageAllocator& allocator,
                          const std::string& model_id) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (allocator.is_suspended(model_id)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

TEST(PagedKVCachePageAllocatorTest,
     VirtualPageExhaustionReturnsNullWithoutChangingCounts) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto first = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto second = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->page_id(), 0);
  EXPECT_EQ(second->page_id(), 1);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 2);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.mapped_offsets(), std::vector<int64_t>({0, 64}));
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {first->page_id()});
  auto reused = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->page_id(), first->page_id());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.mapped_offsets(), std::vector<int64_t>({0, 64}));
}

TEST(PagedKVCachePageAllocatorTest,
     PhysicalPageExhaustionPreservesFreeVirtualPages) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("first", /*num_layers=*/2));
  ASSERT_TRUE(allocator.register_model("second", /*num_layers=*/2));
  ASSERT_NE(allocator.alloc_kv_cache_page("first", /*dp_rank=*/0), nullptr);
  ASSERT_NE(allocator.alloc_kv_cache_page("second", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(allocator.alloc_kv_cache_page("first", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(allocator.get_num_free_virt_pages("first", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("first", /*dp_rank=*/0), 1);
  EXPECT_EQ(backend.available_pages("first", /*dp_rank=*/0), 0);
}

TEST(PagedKVCachePageAllocatorTest, DpGroupsReserveIndependentBackendCapacity) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/2);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  ASSERT_NE(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
  ASSERT_NE(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/1);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 4);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/1, {page->page_id()});
  allocator.trim_kv_cache("model", /*dp_rank=*/1);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 8);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/1), 0);
}

TEST(PagedKVCachePageAllocatorTest,
     ExcessFreedPagesReleaseBackendCapacityAfterUnmapping) {
  FakeKVCachePageBackend backend(/*total_pages=*/80, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/80,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  std::vector<int64_t> page_ids;
  page_ids.reserve(40);
  for (size_t index = 0; index < 40; ++index) {
    auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
    ASSERT_NE(page, nullptr);
    page_ids.emplace_back(page->page_id());
  }
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, page_ids);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 32);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 16);
  allocator.trim_kv_cache("model", /*dp_rank=*/0);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 80);
}

TEST(PagedKVCachePageAllocatorTest,
     FreeSuspendedPageDoesNotReleaseAnotherModelsPages) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("sleeping", /*num_layers=*/2));
  ASSERT_TRUE(allocator.register_model("awake", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("sleeping", /*dp_rank=*/0);
  auto reserved = allocator.alloc_kv_cache_page("sleeping", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  ASSERT_NE(reserved, nullptr);
  allocator.free_kv_cache_pages(
      "sleeping", /*dp_rank=*/0, {reserved->page_id()});
  auto plan = allocator.begin_suspend("sleeping");
  ASSERT_TRUE(plan.has_value());
  complete_suspend(allocator, backend, "sleeping", *plan);
  ASSERT_NE(allocator.alloc_kv_cache_page("awake", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(backend.available_pages("awake", /*dp_rank=*/0), 4);
  allocator.free_kv_cache_pages("sleeping", /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(backend.available_pages("awake", /*dp_rank=*/0), 4);
  EXPECT_EQ(allocator.get_num_free_virt_pages("sleeping", /*dp_rank=*/0), 2);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("sleeping", /*dp_rank=*/0), 0);
  allocator.trim_kv_cache("sleeping", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("awake", /*dp_rank=*/0), 4);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("sleeping", /*dp_rank=*/0),
            0);
}

TEST(PagedKVCachePageAllocatorTest,
     ResumeWaitsForSuspendToDrainPendingMappings) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  backend.block_maps();
  auto allocating = std::async(std::launch::async, [&allocator] {
    return allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  });
  EXPECT_TRUE(backend.wait_until_map_started());
  auto suspending = std::async(std::launch::async, [&allocator] {
    return allocator.begin_suspend("model");
  });
  EXPECT_TRUE(wait_until_suspended(allocator, "model"));
  EXPECT_EQ(suspending.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  std::promise<void> resume_started;
  auto resuming = std::async(std::launch::async, [&allocator, &resume_started] {
    resume_started.set_value();
    return allocator.begin_resume("model");
  });
  resume_started.get_future().wait();
  EXPECT_EQ(resuming.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  backend.release_maps();
  auto page = allocating.get();
  auto suspend_plan = suspending.get();
  EXPECT_NE(page, nullptr);
  if (suspend_plan.has_value()) {
    EXPECT_EQ(suspend_plan->dp_group_page_ids[0].size(), 1);
    complete_suspend(allocator, backend, "model", *suspend_plan);
  }
  auto resume_plan = resuming.get();
  ASSERT_TRUE(suspend_plan.has_value());
  ASSERT_TRUE(resume_plan.has_value());
  EXPECT_EQ(resume_plan->dp_group_page_ids, suspend_plan->dp_group_page_ids);
  ASSERT_TRUE(backend.try_reserve_pages("model",
                                        /*dp_rank=*/0,
                                        /*physical_pages=*/4));
  allocator.map_pages("model", *resume_plan);
  allocator.finish_resume("model", /*success=*/true);
  EXPECT_FALSE(allocator.is_suspended("model"));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 4);
}

TEST(PagedKVCachePageAllocatorTest,
     FreeWaitsForResumeBeforeRecyclingMappedPage) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  auto suspend_plan = allocator.begin_suspend("model");
  ASSERT_TRUE(suspend_plan.has_value());
  complete_suspend(allocator, backend, "model", *suspend_plan);
  auto resume_plan = allocator.begin_resume("model");
  ASSERT_TRUE(resume_plan.has_value());
  ASSERT_TRUE(backend.try_reserve_pages("model",
                                        /*dp_rank=*/0,
                                        /*physical_pages=*/4));
  allocator.map_pages("model", *resume_plan);
  std::promise<void> free_started;
  auto freeing =
      std::async(std::launch::async, [&allocator, &page, &free_started] {
        free_started.set_value();
        allocator.free_kv_cache_pages(
            "model", /*dp_rank=*/0, {page->page_id()});
      });
  free_started.get_future().wait();
  EXPECT_EQ(freeing.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 1);
  allocator.finish_resume("model", /*success=*/true);
  freeing.get();
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 1);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 4);
  auto reused = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(reused, nullptr);
  EXPECT_EQ(reused->page_id(), page->page_id());
}

TEST(PagedKVCachePageAllocatorTest,
     TrimWaitsForSuspendBeforeReleasingVirtualIds) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  auto plan = allocator.begin_suspend("model");
  ASSERT_TRUE(plan.has_value());
  std::promise<void> trim_started;
  auto trimming = std::async(std::launch::async, [&allocator, &trim_started] {
    trim_started.set_value();
    allocator.trim_kv_cache("model", /*dp_rank=*/0);
  });
  trim_started.get_future().wait();
  EXPECT_EQ(trimming.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 1);
  complete_suspend(allocator, backend, "model", *plan);
  trimming.get();
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 8);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 2);
}

TEST(PagedKVCachePageAllocatorTest, FailedResumeKeepsPageIdsSuspendedForRetry) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  auto suspend_plan = allocator.begin_suspend("model");
  ASSERT_TRUE(suspend_plan.has_value());
  complete_suspend(allocator, backend, "model", *suspend_plan);
  EXPECT_FALSE(allocator.begin_suspend("model").has_value());
  auto first_resume = allocator.begin_resume("model");
  ASSERT_TRUE(first_resume.has_value());
  allocator.finish_resume("model", /*success=*/false);
  EXPECT_TRUE(allocator.is_suspended("model"));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 8);
  auto retry = allocator.begin_resume("model");
  ASSERT_TRUE(retry.has_value());
  EXPECT_EQ(retry->dp_group_page_ids, first_resume->dp_group_page_ids);
  allocator.finish_resume("model", /*success=*/false);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 8);
}

TEST(PagedKVCachePageAllocatorTest,
     SuspendDrainsReservedPreallocationBatchesBeforeReturningPlan) {
  FakeKVCachePageBackend backend(/*total_pages=*/16, /*dp_size=*/2);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/16,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  backend.block_maps();
  allocator.start_prealloc_thread();
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 0);
  auto suspending = std::async(std::launch::async, [&allocator] {
    return allocator.begin_suspend("model");
  });
  EXPECT_TRUE(wait_until_suspended(allocator, "model"));
  EXPECT_EQ(suspending.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  backend.release_maps();
  auto plan = suspending.get();
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->dp_group_page_ids[0].size(), 8);
  EXPECT_TRUE(plan->dp_group_page_ids[1].empty());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 16);
  EXPECT_EQ(backend.map_dp_ranks(), std::vector<int32_t>({0}));
  complete_suspend(allocator, backend, "model", *plan);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 16);
}

TEST(PagedKVCachePageAllocatorTest, UsesInjectedPageGeometry) {
  FakeKVCachePageBackend backend(/*total_pages=*/24, /*dp_size=*/1);
  PagedKVCachePageAllocator allocator(backend);
  allocator.init(/*total_pages=*/24,
                 /*page_size=*/128,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/3));
  EXPECT_FALSE(allocator.register_model("model", /*num_layers=*/3));
  EXPECT_EQ(allocator.get_num_total_virt_pages("model"), 4);
  EXPECT_EQ(allocator.phy_pages_per_virt_page("model"), 6);
  EXPECT_EQ(
      allocator.get_virt_page_id(/*block_id=*/7, /*block_memory_size=*/32), 1);
  EXPECT_EQ(allocator.get_offset(/*virtual_page_id=*/3), 384);
}

}  // namespace
}  // namespace xllm

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

#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"

namespace xllm {
namespace {

class AllocationRejectionLogSink final : public google::LogSink {
 public:
  explicit AllocationRejectionLogSink(const std::string& model_id)
      : model_filter_("model_id=" + model_id) {
    google::AddLogSink(this);
  }

  ~AllocationRejectionLogSink() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*logmsgtime*/,
            const char* message,
            size_t message_len) override {
    const std::string_view text(message, message_len);
    if (severity == google::GLOG_WARNING &&
        text.find("reason=physical_page_budget_rejected") !=
            std::string_view::npos &&
        text.find(model_filter_) != std::string_view::npos) {
      ++warning_count_;
    }
  }

  size_t warning_count() const { return warning_count_; }

 private:
  const std::string model_filter_;
  size_t warning_count_ = 0;
};

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
    CHECK(!forbid_budget_release_)
        << "Physical budget released before unmapping completed";
    EXPECT_LE(physical_pages, total_pages_ - free_pages_[dp_rank]);
    free_pages_[dp_rank] += physical_pages;
    cond_.notify_all();
  }

  size_t available_pages(const std::string& /*model_id*/,
                         int32_t dp_rank) const override {
    std::lock_guard<std::mutex> lock(mtx_);
    return free_pages_[dp_rank];
  }

  bool map_pages(const std::string& model_id,
                 int32_t dp_rank,
                 const std::vector<int64_t>& byte_offsets) override {
    std::unique_lock<std::mutex> lock(mtx_);
    mapped_model_ids_.emplace_back(model_id);
    map_dp_ranks_.emplace_back(dp_rank);
    mapped_offsets_.insert(
        mapped_offsets_.end(), byte_offsets.begin(), byte_offsets.end());
    const bool success = map_dp_ranks_.size() != failed_map_call_;
    map_started_ = true;
    cond_.notify_all();
    cond_.wait(lock, [this] { return !block_maps_; });
    return success;
  }

  bool unmap_pages(const std::string& /*model_id*/,
                   int32_t dp_rank,
                   const std::vector<int64_t>& byte_offsets) override {
    std::lock_guard<std::mutex> lock(mtx_);
    unmap_dp_ranks_.emplace_back(dp_rank);
    unmapped_offsets_.insert(
        unmapped_offsets_.end(), byte_offsets.begin(), byte_offsets.end());
    return unmap_dp_ranks_.size() != failed_unmap_call_;
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

  void fail_map_call(size_t call) {
    std::lock_guard<std::mutex> lock(mtx_);
    failed_map_call_ = call;
  }

  void fail_unmap_call(size_t call) {
    std::lock_guard<std::mutex> lock(mtx_);
    failed_unmap_call_ = call;
  }

  void forbid_budget_release() {
    std::lock_guard<std::mutex> lock(mtx_);
    forbid_budget_release_ = true;
  }

  bool wait_until_available_pages(int32_t dp_rank, size_t physical_pages) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cond_.wait_for(
        lock, std::chrono::seconds(2), [this, dp_rank, physical_pages] {
          return free_pages_[dp_rank] == physical_pages;
        });
  }

  std::vector<int64_t> mapped_offsets() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return mapped_offsets_;
  }

  std::vector<int32_t> map_dp_ranks() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return map_dp_ranks_;
  }

  std::vector<int64_t> unmapped_offsets() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return unmapped_offsets_;
  }

  std::vector<int32_t> unmap_dp_ranks() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return unmap_dp_ranks_;
  }

  std::vector<std::string> mapped_model_ids() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return mapped_model_ids_;
  }

 private:
  size_t total_pages_;
  std::vector<size_t> free_pages_;
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  bool block_maps_ = false;
  bool map_started_ = false;
  size_t failed_map_call_ = 0;
  size_t failed_unmap_call_ = 0;
  bool forbid_budget_release_ = false;
  std::vector<int64_t> mapped_offsets_;
  std::vector<int64_t> unmapped_offsets_;
  std::vector<int32_t> map_dp_ranks_;
  std::vector<int32_t> unmap_dp_ranks_;
  std::vector<std::string> mapped_model_ids_;
};

// Emulate the residency coordinator's completed lifecycle budget transaction.
void complete_suspend(VmmKVPageAllocatorImpl& allocator,
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

bool wait_until_suspended(const VmmKVPageAllocatorImpl& allocator,
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

bool wait_until_unavailable(const VmmKVPageAllocatorImpl& allocator,
                            const std::string& model_id) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (allocator.get_num_available_virt_pages(model_id, /*dp_rank=*/0) == 0) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

TEST(VmmKVPageAllocatorImplTest,
     VirtualPageExhaustionReturnsNullWithoutChangingCounts) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest,
     LargeVirtualBudgetRegistersWithoutMaterializingEveryPageId) {
  constexpr size_t kVirtualPages = size_t{1} << 40;
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(kVirtualPages * 2,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false,
                 /*max_reserved_pages=*/0);

  ASSERT_TRUE(allocator.register_model("large", /*num_layers=*/1));
  EXPECT_EQ(allocator.get_num_total_virt_pages("large"), kVirtualPages);
  EXPECT_EQ(allocator.get_num_free_virt_pages("large", /*dp_rank=*/0),
            kVirtualPages);

  auto first = allocator.alloc_kv_cache_page("large", /*dp_rank=*/0);
  auto second = allocator.alloc_kv_cache_page("large", /*dp_rank=*/0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->page_id(), 0);
  EXPECT_EQ(second->page_id(), 1);
}

TEST(VmmKVPageAllocatorImplTest,
     PhysicalBudgetRejectionsLimitWarningsAndRecover) {
  const std::string model_id = "budget-log-rate-limit";
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false,
                 /*max_reserved_pages=*/0);
  ASSERT_TRUE(allocator.register_model(model_id, /*num_layers=*/2));
  ASSERT_TRUE(backend.try_reserve_pages("weights", /*dp_rank=*/0, 8));
  AllocationRejectionLogSink log_sink(model_id);

  for (size_t attempt = 0; attempt < 2000; ++attempt) {
    EXPECT_EQ(allocator.alloc_kv_cache_page(model_id, /*dp_rank=*/0), nullptr);
  }
  EXPECT_GT(log_sink.warning_count(), 0u);
  EXPECT_LE(log_sink.warning_count(), 3u);
  EXPECT_EQ(allocator.get_num_free_virt_pages(model_id, /*dp_rank=*/0), 2u);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages(model_id, /*dp_rank=*/0), 0u);
  EXPECT_TRUE(backend.mapped_offsets().empty());

  backend.release_pages("weights", /*dp_rank=*/0, 8);
  auto page = allocator.alloc_kv_cache_page(model_id, /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(page->page_id(), 0);
  EXPECT_EQ(backend.available_pages(model_id, /*dp_rank=*/0), 4u);
  allocator.free_kv_cache_pages(model_id, /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(backend.available_pages(model_id, /*dp_rank=*/0), 8u);
}

TEST(VmmKVPageAllocatorImplTest,
     FreshVirtualIdsPrecedeRecycledIdsAndThenRecycleInFifoOrder) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false,
                 /*max_reserved_pages=*/0);
  ASSERT_TRUE(allocator.register_model("model",
                                       /*num_layers=*/1,
                                       /*physical_pages_per_virtual_page=*/0,
                                       /*max_virtual_pages=*/4));

  auto page_zero = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto page_one = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page_zero, nullptr);
  ASSERT_NE(page_one, nullptr);
  EXPECT_EQ(page_zero->page_id(), 0);
  EXPECT_EQ(page_one->page_id(), 1);

  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page_zero->page_id()});
  auto fresh_two = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(fresh_two, nullptr);
  EXPECT_EQ(fresh_two->page_id(), 2);

  allocator.free_kv_cache_pages(
      "model", /*dp_rank=*/0, {page_one->page_id(), fresh_two->page_id()});
  auto fresh_three = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(fresh_three, nullptr);
  EXPECT_EQ(fresh_three->page_id(), 3);

  allocator.free_kv_cache_pages(
      "model", /*dp_rank=*/0, {fresh_three->page_id()});
  backend.fail_map_call(5);
  EXPECT_EQ(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 4);
  auto recycled_zero = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto recycled_one = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(recycled_zero, nullptr);
  ASSERT_NE(recycled_one, nullptr);
  EXPECT_EQ(recycled_zero->page_id(), 0);
  EXPECT_EQ(recycled_one->page_id(), 1);
}

TEST(VmmKVPageAllocatorImplTest,
     TrimmedVirtualIdsFollowFreshIdsWithoutIncreasingLogicalCapacity) {
  FakeKVCachePageBackend backend(/*total_pages=*/6, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/6,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 3);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 1);

  allocator.trim_kv_cache("model", /*dp_rank=*/0);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 3);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 3);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 6);

  auto first = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto second = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  auto recycled = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(recycled, nullptr);
  EXPECT_EQ(first->page_id(), 1);
  EXPECT_EQ(second->page_id(), 2);
  EXPECT_EQ(recycled->page_id(), 0);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
}

TEST(VmmKVPageAllocatorImplTest,
     FailedMappingRestoresVirtualIdAndPhysicalBudgetForRetry) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  backend.fail_map_call(1);

  EXPECT_EQ(allocator.alloc_kv_cache_page("model", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 2);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 8);

  auto retry = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(retry, nullptr);
  EXPECT_EQ(retry->page_id(), 0);
  EXPECT_EQ(backend.mapped_offsets(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 4);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 1);
  auto fresh = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(fresh, nullptr);
  EXPECT_EQ(fresh->page_id(), 1);
}

TEST(VmmKVPageAllocatorImplTest,
     FailedPreallocationRestoresBudgetWithoutPublishingReservedPages) {
  FakeKVCachePageBackend backend(/*total_pages=*/16, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/16,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  backend.fail_map_call(1);
  backend.block_maps();
  allocator.start_prealloc_thread("model");
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  backend.release_maps();
  EXPECT_TRUE(backend.wait_until_available_pages(/*dp_rank=*/0, 16));

  // Suspend drains the failed batch and prevents another background attempt
  // from hiding a leaked reservation or virtual ID.
  auto plan = allocator.begin_suspend("model");
  ASSERT_TRUE(plan.has_value());
  EXPECT_TRUE(plan->dp_group_page_ids[0].empty());
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 8);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(backend.map_dp_ranks(), std::vector<int32_t>({0}));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 16);
  complete_suspend(allocator, backend, "model", *plan);
}

TEST(VmmKVPageAllocatorImplTest,
     FailedMixedPreallocationRetriesWholeBatchBeforeRemainingRecycledIds) {
  FakeKVCachePageBackend backend(/*total_pages=*/20, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/20,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  std::vector<int64_t> page_ids;
  for (size_t index = 0; index < 4; ++index) {
    auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
    ASSERT_NE(page, nullptr);
    page_ids.emplace_back(page->page_id());
  }
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, page_ids);
  allocator.trim_kv_cache("model", /*dp_rank=*/0);
  backend.fail_map_call(5);
  backend.block_maps();
  allocator.start_prealloc_thread("model");
  EXPECT_TRUE(backend.wait_until_map_started());
  backend.release_maps();
  EXPECT_TRUE(backend.wait_until_available_pages(/*dp_rank=*/0, 20));
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 10);

  // The failed batch spans the last fresh IDs and the first recycled IDs.
  // Its retry must precede recycled IDs 2 and 3 and retain the entire order.
  backend.block_maps();
  allocator.start_prealloc_thread("model");
  EXPECT_TRUE(backend.wait_until_map_started());
  auto suspending = std::async(std::launch::async, [&allocator] {
    return allocator.begin_suspend("model");
  });
  EXPECT_TRUE(wait_until_suspended(allocator, "model"));
  backend.release_maps();
  auto plan = suspending.get();
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->dp_group_page_ids[0],
            (std::vector<int64_t>{4, 5, 6, 7, 8, 9, 0, 1}));
  EXPECT_EQ(
      backend.mapped_offsets(),
      (std::vector<int64_t>{0, 64, 128, 192, 256, 320, 384, 448, 512, 576,
                            0, 64, 256, 320, 384, 448, 512, 576, 0,   64}));
  complete_suspend(allocator, backend, "model", *plan);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 20);
}

TEST(VmmKVPageAllocatorImplTest,
     FailedResumeMappingRollsBackEarlierDpGroupsBeforeBudgetRelease) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/3);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/3,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  for (int32_t dp_rank = 0; dp_rank < 3; ++dp_rank) {
    ASSERT_NE(allocator.alloc_kv_cache_page("model", dp_rank), nullptr);
  }
  auto suspend_plan = allocator.begin_suspend("model");
  ASSERT_TRUE(suspend_plan.has_value());
  complete_suspend(allocator, backend, "model", *suspend_plan);
  auto resume_plan = allocator.begin_resume("model");
  ASSERT_TRUE(resume_plan.has_value());
  for (int32_t dp_rank = 0; dp_rank < 3; ++dp_rank) {
    ASSERT_TRUE(backend.try_reserve_pages("model", dp_rank, 4));
  }

  // The first three maps allocated the original pages; fail the second
  // resume group and require the first group to be unmapped again.
  backend.fail_map_call(5);
  EXPECT_FALSE(allocator.map_pages("model", *resume_plan));
  EXPECT_EQ(backend.map_dp_ranks(), std::vector<int32_t>({0, 1, 2, 0, 1}));
  EXPECT_EQ(backend.unmap_dp_ranks(), std::vector<int32_t>({0, 1, 2, 0}));
  EXPECT_EQ(backend.unmapped_offsets(), std::vector<int64_t>({0, 0, 0, 0}));
  for (int32_t dp_rank = 0; dp_rank < 3; ++dp_rank) {
    EXPECT_EQ(backend.available_pages("model", dp_rank), 4);
    backend.release_pages("model", dp_rank, 4);
  }
  allocator.finish_resume("model", /*success=*/false);
  EXPECT_TRUE(allocator.is_suspended("model"));

  auto retry = allocator.begin_resume("model");
  ASSERT_TRUE(retry.has_value());
  EXPECT_EQ(retry->dp_group_page_ids, resume_plan->dp_group_page_ids);
  for (int32_t dp_rank = 0; dp_rank < 3; ++dp_rank) {
    ASSERT_TRUE(backend.try_reserve_pages("model", dp_rank, 4));
  }
  ASSERT_TRUE(allocator.map_pages("model", *retry));
  allocator.finish_resume("model", /*success=*/true);
  EXPECT_FALSE(allocator.is_suspended("model"));
  for (int32_t dp_rank = 0; dp_rank < 3; ++dp_rank) {
    EXPECT_EQ(backend.available_pages("model", dp_rank), 4);
    EXPECT_EQ(allocator.get_num_inuse_virt_pages("model", dp_rank), 1);
  }
}

TEST(VmmKVPageAllocatorImplTest,
     AvailablePagesFollowLiveBudgetAndResidentReserve) {
  FakeKVCachePageBackend backend(/*total_pages=*/12, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/12,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 3);
  ASSERT_TRUE(backend.try_reserve_pages("weights", /*dp_rank=*/0, 5));
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 1);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 3);
  backend.release_pages("weights", /*dp_rank=*/0, 5);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 3);

  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  ASSERT_TRUE(backend.try_reserve_pages("weights", /*dp_rank=*/0, 8));
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 1);
  auto resident = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(resident, nullptr);
  EXPECT_EQ(resident->page_id(), page->page_id());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
  backend.release_pages("weights", /*dp_rank=*/0, 8);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 2);

  auto plan = allocator.begin_suspend("model");
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
  complete_suspend(allocator, backend, "model", *plan);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 12);
  EXPECT_EQ(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
}

TEST(VmmKVPageAllocatorImplTest, ZeroReserveUnmapsAndReleasesEveryFreedPage) {
  FakeKVCachePageBackend backend(/*total_pages=*/4, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/4,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/true,
                 /*max_reserved_pages=*/0);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  allocator.start_prealloc_thread("model");
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 0);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  EXPECT_EQ(backend.unmapped_offsets(), std::vector<int64_t>({0}));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 4);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("model", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.get_num_free_virt_pages("model", /*dp_rank=*/0), 1);
  auto retry = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(retry, nullptr);
  EXPECT_EQ(retry->page_id(), page->page_id());
  EXPECT_EQ(backend.mapped_offsets(), std::vector<int64_t>({0, 0}));
}

TEST(VmmKVPageAllocatorImplTest,
     PhysicalPageExhaustionPreservesFreeVirtualPages) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, DpGroupsReserveIndependentBackendCapacity) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest,
     ExcessFreedPagesReleaseBackendCapacityAfterUnmapping) {
  FakeKVCachePageBackend backend(/*total_pages=*/80, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest,
     FreeSuspendedPageDoesNotReleaseAnotherModelsPages) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, ResumeWaitsForSuspendToDrainPendingMappings) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, FreeWaitsForResumeBeforeRecyclingMappedPage) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, TrimWaitsForSuspendBeforeReleasingVirtualIds) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, FailedResumeKeepsPageIdsSuspendedForRetry) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest,
     SuspendDrainsReservedPreallocationBatchesBeforeReturningPlan) {
  FakeKVCachePageBackend backend(/*total_pages=*/16, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/16,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  backend.block_maps();
  allocator.start_prealloc_thread("model");
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

TEST(VmmKVPageAllocatorImplTest,
     UnregisterDrainsExecutingAndQueuedPreallocationBeforeReregister) {
  FakeKVCachePageBackend backend(/*total_pages=*/32, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/32,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  backend.block_maps();
  allocator.start_prealloc_thread("model");
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 16);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 16);
  EXPECT_GT(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
  std::promise<void> unregister_started;
  auto unregistering =
      std::async(std::launch::async, [&allocator, &unregister_started] {
        unregister_started.set_value();
        return allocator.unregister_model("model");
      });
  unregister_started.get_future().wait();
  EXPECT_TRUE(wait_until_unavailable(allocator, "model"));
  EXPECT_EQ(unregistering.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  backend.release_maps();
  ASSERT_TRUE(unregistering.get());
  EXPECT_EQ(backend.map_dp_ranks(), (std::vector<int32_t>{0}));
  EXPECT_EQ(backend.unmap_dp_ranks(), (std::vector<int32_t>{0}));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 32);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 32);

  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  EXPECT_EQ(allocator.phy_pages_per_virt_page("model"), 4);
  EXPECT_EQ(allocator.get_num_total_virt_pages("model"), 8);
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/1);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(page->page_id(), 0);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 28);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/1, {page->page_id()});
  page.reset();
  EXPECT_TRUE(allocator.unregister_model("model"));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 32);
}

TEST(VmmKVPageAllocatorImplTest,
     UnregisterRejectsLiveAllocationsAndUnfinishedTransitions) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  EXPECT_FALSE(allocator.unregister_model("model"));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 6);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  page.reset();
  auto suspend_plan = allocator.begin_suspend("model");
  ASSERT_TRUE(suspend_plan.has_value());
  EXPECT_FALSE(allocator.unregister_model("model"));
  complete_suspend(allocator, backend, "model", *suspend_plan);
  auto resume_plan = allocator.begin_resume("model");
  ASSERT_TRUE(resume_plan.has_value());
  EXPECT_FALSE(allocator.unregister_model("model"));
  allocator.finish_resume("model", /*success=*/false);
  EXPECT_TRUE(allocator.unregister_model("model"));
  EXPECT_FALSE(allocator.unregister_model("model"));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 8);
}

TEST(VmmKVPageAllocatorImplTest,
     UnregisterSuspendedModelDoesNotReleaseItsBudgetTwice) {
  FakeKVCachePageBackend backend(/*total_pages=*/8, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/8,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  ASSERT_TRUE(allocator.register_model("other", /*num_layers=*/1));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  page.reset();
  auto plan = allocator.begin_suspend("model");
  ASSERT_TRUE(plan.has_value());
  complete_suspend(allocator, backend, "model", *plan);
  auto other_page = allocator.alloc_kv_cache_page("other", /*dp_rank=*/0);
  ASSERT_NE(other_page, nullptr);
  EXPECT_TRUE(allocator.unregister_model("model"));
  EXPECT_EQ(backend.unmapped_offsets(), (std::vector<int64_t>{0}));
  EXPECT_EQ(backend.available_pages("other", /*dp_rank=*/0), 6);
  EXPECT_EQ(allocator.get_num_inuse_virt_pages("other", /*dp_rank=*/0), 1);
  EXPECT_TRUE(allocator.register_model("model", /*num_layers=*/2));
}

TEST(VmmKVPageAllocatorImplTest,
     ShutdownDrainsExecutingAndQueuedPreallocationBeforeBackendTeardown) {
  FakeKVCachePageBackend backend(/*total_pages=*/32, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/32,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/1));
  backend.block_maps();
  allocator.start_prealloc_thread("model");
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_GT(allocator.get_num_available_virt_pages("model", /*dp_rank=*/0), 0);
  std::promise<void> shutdown_started;
  auto shutdown =
      std::async(std::launch::async, [&allocator, &shutdown_started] {
        shutdown_started.set_value();
        allocator.shutdown();
      });
  shutdown_started.get_future().wait();
  EXPECT_TRUE(wait_until_unavailable(allocator, "model"));
  EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  backend.release_maps();
  shutdown.get();
  EXPECT_EQ(backend.map_dp_ranks(), (std::vector<int32_t>{0}));
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/0), 16);
  EXPECT_EQ(backend.available_pages("model", /*dp_rank=*/1), 32);
  // Resident mappings remain owned until the backend's resource teardown.
  EXPECT_TRUE(backend.unmapped_offsets().empty());
}

TEST(VmmKVPageAllocatorImplTest, UsesInjectedPageGeometry) {
  FakeKVCachePageBackend backend(/*total_pages=*/24, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
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

TEST(VmmKVPageAllocatorImplTest, PreallocationWaitsForEachModelsWorkerRegions) {
  FakeKVCachePageBackend backend(/*total_pages=*/64, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/64,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("first", /*num_layers=*/1));
  ASSERT_TRUE(allocator.register_model("second", /*num_layers=*/1));
  backend.block_maps();
  allocator.start_prealloc_thread("first");
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_EQ(backend.mapped_model_ids(), std::vector<std::string>({"first"}));
  EXPECT_EQ(backend.available_pages("first", /*dp_rank=*/0), 48);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("second", /*dp_rank=*/0), 0);
  backend.release_maps();
  auto first_plan = allocator.begin_suspend("first");
  ASSERT_TRUE(first_plan.has_value());
  EXPECT_EQ(first_plan->dp_group_page_ids[0].size(), 8);
  complete_suspend(allocator, backend, "first", *first_plan);
  EXPECT_EQ(backend.available_pages("second", /*dp_rank=*/0), 64);

  backend.block_maps();
  allocator.start_prealloc_thread("second");
  EXPECT_TRUE(backend.wait_until_map_started());
  EXPECT_EQ(backend.mapped_model_ids(),
            std::vector<std::string>({"first", "second"}));
  EXPECT_EQ(backend.available_pages("second", /*dp_rank=*/0), 48);
  backend.release_maps();
  auto second_plan = allocator.begin_suspend("second");
  ASSERT_TRUE(second_plan.has_value());
  EXPECT_EQ(second_plan->dp_group_page_ids[0].size(), 8);
  complete_suspend(allocator, backend, "second", *second_plan);
  EXPECT_EQ(backend.available_pages("second", /*dp_rank=*/0), 64);
}

TEST(VmmKVPageAllocatorImplTest, MlaBudgetSurvivesTrimAndSleepWakeup) {
  FakeKVCachePageBackend backend(/*total_pages=*/36, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/36,
                 /*page_size=*/2 * 1024 * 1024,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("mla",
                                       /*num_layers=*/2,
                                       /*physical_pages_per_virtual_page=*/18,
                                       /*max_virtual_pages=*/2));
  EXPECT_EQ(allocator.phy_pages_per_virt_page("mla"), 18);
  auto first = allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0);
  auto second = allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 0);
  EXPECT_EQ(allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0), nullptr);
  allocator.free_kv_cache_pages("mla", /*dp_rank=*/0, {first->page_id()});
  allocator.trim_kv_cache("mla", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 18);
  auto plan = allocator.begin_suspend("mla");
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->physical_pages_per_virtual_page, 18);
  EXPECT_EQ(plan->dp_group_page_ids[0],
            std::vector<int64_t>({second->page_id()}));
  complete_suspend(allocator, backend, "mla", *plan);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 36);
  auto resume = allocator.begin_resume("mla");
  ASSERT_TRUE(resume.has_value());
  ASSERT_TRUE(backend.try_reserve_pages(
      "mla", /*dp_rank=*/0, resume->physical_pages_per_virtual_page));
  allocator.map_pages("mla", *resume);
  allocator.finish_resume("mla", /*success=*/true);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 18);
  allocator.free_kv_cache_pages("mla", /*dp_rank=*/0, {second->page_id()});
  allocator.trim_kv_cache("mla", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 36);
}

TEST(VmmKVPageAllocatorImplTest, MlaShortagePreservesPhysicalBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/36, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/36,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("mla",
                                       /*num_layers=*/2,
                                       /*physical_pages_per_virtual_page=*/18));
  ASSERT_TRUE(backend.try_reserve_pages("weights", /*dp_rank=*/0, 19));
  EXPECT_EQ(allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 17);
  EXPECT_EQ(allocator.get_num_free_virt_pages("mla", /*dp_rank=*/0), 2);
  backend.release_pages("weights", /*dp_rank=*/0, 1);
  EXPECT_NE(allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 0);
}

TEST(VmmKVPageAllocatorImplTest, TensorCapacityLimitsVirtualPageIds) {
  for (size_t blocks : {size_t{128}, size_t{129}, size_t{200}, size_t{256}}) {
    FakeKVCachePageBackend backend(/*total_pages=*/360, /*dp_size=*/1);
    VmmKVPageAllocatorImpl allocator(backend);
    allocator.init(/*total_pages=*/360,
                   /*page_size=*/2 * 1024 * 1024,
                   /*dp_size=*/1,
                   /*enable_prealloc=*/false);
    ASSERT_TRUE(allocator.register_model("mla",
                                         /*num_layers=*/2,
                                         /*physical_pages_per_virtual_page=*/18,
                                         /*max_virtual_pages=*/blocks / 128));
    EXPECT_EQ(allocator.get_num_total_virt_pages("mla"), blocks / 128);
    for (size_t index = 0; index < blocks / 128; ++index) {
      auto page = allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0);
      ASSERT_NE(page, nullptr);
      page->init(/*block_mem_size=*/16 * 1024);
      const auto ids = page->alloc(128);
      EXPECT_LT(ids.back(), blocks);
    }
    EXPECT_EQ(allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0), nullptr);
  }
}

TEST(VmmKVPageAllocatorImplTest, MlaPreallocationKeepsPhysicalPageBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/180, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/180,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/true);
  ASSERT_TRUE(allocator.register_model("mla",
                                       /*num_layers=*/2,
                                       /*physical_pages_per_virtual_page=*/18));
  allocator.start_prealloc_thread("mla");
  ASSERT_TRUE(backend.wait_until_map_started());
  auto plan = allocator.begin_suspend("mla");
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->dp_group_page_ids[0].size(), 1);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 162);
  complete_suspend(allocator, backend, "mla", *plan);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 180);
}

TEST(VmmKVPageAllocatorImplTest, MlaRetainedPagesKeepPhysicalPageBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/180, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/180,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("mla",
                                       /*num_layers=*/2,
                                       /*physical_pages_per_virtual_page=*/18));
  std::vector<int64_t> ids;
  ids.reserve(10);
  for (size_t index = 0; index < 10; ++index) {
    auto page = allocator.alloc_kv_cache_page("mla", /*dp_rank=*/0);
    ASSERT_NE(page, nullptr);
    ids.emplace_back(page->page_id());
  }
  allocator.free_kv_cache_pages("mla", /*dp_rank=*/0, ids);
  EXPECT_EQ(allocator.get_num_reserved_virt_pages("mla", /*dp_rank=*/0), 7);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 54);
  allocator.trim_kv_cache("mla", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("mla", /*dp_rank=*/0), 180);
}

TEST(VmmKVPageAllocatorImplTest,
     Glm53IndexerBudgetSurvivesShortageAndSleepWakeup) {
  const VirtualKVCachePageLayout layout({128 * 1024, 16 * 1024, 32 * 1024},
                                        /*page_size=*/2 * 1024 * 1024);
  const size_t physical_pages = layout.physical_pages({78, 78, 21});
  FakeKVCachePageBackend backend(/*total_pages=*/2 * physical_pages,
                                 /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/2 * physical_pages,
                 /*page_size=*/2 * 1024 * 1024,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("glm53",
                                       /*num_layers=*/78,
                                       physical_pages,
                                       /*max_virtual_pages=*/2));
  EXPECT_EQ(allocator.phy_pages_per_virt_page("glm53"), 744);
  ASSERT_TRUE(backend.try_reserve_pages("weights", /*dp_rank=*/0, 1));
  auto page = allocator.alloc_kv_cache_page("glm53", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(allocator.alloc_kv_cache_page("glm53", /*dp_rank=*/0), nullptr);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 743);
  EXPECT_EQ(allocator.get_num_free_virt_pages("glm53", /*dp_rank=*/0), 1);
  backend.release_pages("weights", /*dp_rank=*/0, 1);
  auto second = allocator.alloc_kv_cache_page("glm53", /*dp_rank=*/0);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 0);
  allocator.free_kv_cache_pages("glm53", /*dp_rank=*/0, {second->page_id()});
  allocator.trim_kv_cache("glm53", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 744);
  auto plan = allocator.begin_suspend("glm53");
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->physical_pages_per_virtual_page, 744);
  EXPECT_EQ(plan->dp_group_page_ids[0],
            std::vector<int64_t>({page->page_id()}));
  complete_suspend(allocator, backend, "glm53", *plan);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 1488);
  auto resume = allocator.begin_resume("glm53");
  ASSERT_TRUE(resume.has_value());
  ASSERT_TRUE(backend.try_reserve_pages(
      "glm53", /*dp_rank=*/0, resume->physical_pages_per_virtual_page));
  allocator.map_pages("glm53", *resume);
  allocator.finish_resume("glm53", /*success=*/true);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 744);
  allocator.free_kv_cache_pages("glm53", /*dp_rank=*/0, {page->page_id()});
  allocator.trim_kv_cache("glm53", /*dp_rank=*/0);
  EXPECT_EQ(backend.available_pages("glm53", /*dp_rank=*/0), 1488);
}

TEST(VmmKVPageAllocatorImplDeathTest,
     FailedFreeUnmapTerminatesBeforeReleasingBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/4, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/4,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false,
                 /*max_reserved_pages=*/0);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  backend.fail_unmap_call(1);
  backend.forbid_budget_release();

  EXPECT_DEATH(
      allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()}),
      "Failed to unmap KV pages for model=model dp_rank=0");
}

TEST(VmmKVPageAllocatorImplDeathTest,
     FailedTrimUnmapTerminatesBeforeReleasingBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/4, /*dp_size=*/1);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/4,
                 /*page_size=*/64,
                 /*dp_size=*/1,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  auto page = allocator.alloc_kv_cache_page("model", /*dp_rank=*/0);
  ASSERT_NE(page, nullptr);
  allocator.free_kv_cache_pages("model", /*dp_rank=*/0, {page->page_id()});
  backend.fail_unmap_call(1);
  backend.forbid_budget_release();

  EXPECT_DEATH(allocator.trim_kv_cache("model", /*dp_rank=*/0),
               "Failed to unmap KV pages for model=model dp_rank=0");
}

TEST(VmmKVPageAllocatorImplDeathTest,
     PartialSuspendUnmapTerminatesBeforeReleasingAnyDpBudget) {
  FakeKVCachePageBackend backend(/*total_pages=*/4, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/4,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
    ASSERT_NE(allocator.alloc_kv_cache_page("model", dp_rank), nullptr);
  }
  auto plan = allocator.begin_suspend("model");
  ASSERT_TRUE(plan.has_value());
  backend.fail_unmap_call(2);
  backend.forbid_budget_release();

  EXPECT_DEATH(complete_suspend(allocator, backend, "model", *plan),
               "Failed to unmap KV pages for model=model dp_rank=1");
}

TEST(VmmKVPageAllocatorImplDeathTest,
     FailedResumeRollbackUnmapTerminatesBeforeReturningFailure) {
  FakeKVCachePageBackend backend(/*total_pages=*/4, /*dp_size=*/2);
  VmmKVPageAllocatorImpl allocator(backend);
  allocator.init(/*total_pages=*/4,
                 /*page_size=*/64,
                 /*dp_size=*/2,
                 /*enable_prealloc=*/false);
  ASSERT_TRUE(allocator.register_model("model", /*num_layers=*/2));
  for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
    ASSERT_NE(allocator.alloc_kv_cache_page("model", dp_rank), nullptr);
  }
  auto suspend_plan = allocator.begin_suspend("model");
  ASSERT_TRUE(suspend_plan.has_value());
  complete_suspend(allocator, backend, "model", *suspend_plan);
  auto resume_plan = allocator.begin_resume("model");
  ASSERT_TRUE(resume_plan.has_value());
  for (int32_t dp_rank = 0; dp_rank < 2; ++dp_rank) {
    ASSERT_TRUE(backend.try_reserve_pages("model", dp_rank, 4));
  }
  // Fail the second resume group, then fail unmapping the first group.
  backend.fail_map_call(4);
  backend.fail_unmap_call(3);
  backend.forbid_budget_release();

  EXPECT_DEATH(allocator.map_pages("model", *resume_plan),
               "Failed to unmap KV pages for model=model dp_rank=0");
}

}  // namespace
}  // namespace xllm

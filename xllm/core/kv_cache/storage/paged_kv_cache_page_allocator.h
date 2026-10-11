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

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/kv_cache/storage/kv_cache_page_allocator.h"
#include "core/kv_cache/storage/kv_cache_page_backend.h"

namespace xllm {

// A stable snapshot of the virtual pages retained across a model transition.
// The residency coordinator reserves/releases the combined KV and weight
// capacity before invoking the mapping helpers below.
struct KVCacheMappingPlan {
  size_t physical_pages_per_virtual_page = 0;
  std::vector<std::vector<int64_t>> dp_group_page_ids;
};

// Owns KV page queues and preallocation policy. Physical page capacity and
// mapping operations are supplied by a backend whose lifetime exceeds this
// allocator's. Model lifecycle and weight policy belong to the caller.
class PagedKVCachePageAllocator final : public KVCachePageAllocator {
 public:
  explicit PagedKVCachePageAllocator(KVCachePageBackend& backend);
  ~PagedKVCachePageAllocator() override;
  PagedKVCachePageAllocator(const PagedKVCachePageAllocator&) = delete;
  PagedKVCachePageAllocator& operator=(const PagedKVCachePageAllocator&) =
      delete;

  void init(size_t total_pages,
            size_t page_size,
            int32_t dp_size,
            bool enable_prealloc = true);
  bool is_initialized() const override;
  bool register_model(const std::string& model_id, int64_t num_layers);
  void start_prealloc_thread() override;

  std::unique_ptr<KVCachePageState> alloc_kv_cache_page(
      const std::string& model_id,
      int32_t dp_rank) override;
  void free_kv_cache_pages(
      const std::string& model_id,
      int32_t dp_rank,
      const std::vector<int64_t>& virtual_page_ids) override;
  void trim_kv_cache(const std::string& model_id, int32_t dp_rank) override;

  // begin_* freezes allocation, recycling and preallocation until finish_*.
  // Suspend also drains every reserved or executing mapping batch before
  // returning its snapshot. A null plan means no transition is required.
  std::optional<KVCacheMappingPlan> begin_suspend(const std::string& model_id);
  void finish_suspend(const std::string& model_id);
  std::optional<KVCacheMappingPlan> begin_resume(const std::string& model_id);
  void finish_resume(const std::string& model_id, bool success);
  bool is_suspended(const std::string& model_id) const;

  // Execute mappings only; the caller owns lifecycle budget transactions.
  void map_pages(const std::string& model_id, const KVCacheMappingPlan& plan);
  void unmap_pages(const std::string& model_id, const KVCacheMappingPlan& plan);

  size_t get_num_free_virt_pages(const std::string& model_id,
                                 int32_t dp_rank) const;
  size_t get_num_inuse_virt_pages(const std::string& model_id,
                                  int32_t dp_rank) const override;
  size_t get_num_reserved_virt_pages(const std::string& model_id,
                                     int32_t dp_rank) const override;
  size_t get_num_total_virt_pages(const std::string& model_id) const;
  int64_t get_virt_page_id(int64_t block_id,
                           size_t block_memory_size) const override;
  int64_t get_offset(int64_t virtual_page_id) const;
  size_t page_size() const override { return page_size_; }
  int64_t num_layers(const std::string& model_id) const;
  size_t phy_pages_per_virt_page(const std::string& model_id) const override;

 private:
  static constexpr size_t kMinReservedPages = 8;
  static constexpr size_t kMaxReservedPages = 32;

  struct DpGroupPages {
    size_t num_free_virt_pages = 0;
    std::deque<int64_t> free_virt_page_list;
    std::deque<int64_t> reserved_virt_page_list;
    std::unordered_set<int64_t> allocated_virt_page_list;
  };

  enum class ModelTransition : uint8_t { IDLE, SUSPENDING, RESUMING };

  struct ModelState {
    int64_t num_layers = 0;
    size_t num_total_virt_pages = 0;
    size_t phy_pages_per_virt_page = 0;
    bool suspended = false;
    bool kv_cache_mapped = true;
    ModelTransition transition = ModelTransition::IDLE;
    size_t pending_map_ops = 0;
    std::vector<DpGroupPages> dp_group_pages;
  };

  struct PreallocationBatch {
    std::string model_id;
    int32_t dp_rank = 0;
    std::vector<int64_t> virtual_page_ids;
  };

  ModelState& get_model_state(const std::string& model_id);
  const ModelState& get_model_state(const std::string& model_id) const;
  void wait_for_model_transition(ModelState& state,
                                 std::unique_lock<std::mutex>& lock);
  KVCacheMappingPlan snapshot_pages(const ModelState& state) const;
  void validate_dp_rank(int32_t dp_rank) const;
  void map_virtual_pages(const std::string& model_id,
                         int32_t dp_rank,
                         const std::vector<int64_t>& virtual_page_ids);
  void unmap_virtual_pages(const std::string& model_id,
                           int32_t dp_rank,
                           const std::vector<int64_t>& virtual_page_ids);
  void prealloc_worker();
  void trigger_preallocation();

  KVCachePageBackend& backend_;
  bool initialized_ = false;
  size_t total_pages_ = 0;
  size_t page_size_ = 0;
  int32_t dp_size_ = 1;
  bool enable_prealloc_ = true;
  std::unordered_map<std::string, ModelState> model_states_;
  mutable std::mutex mtx_;
  std::condition_variable cond_;
  bool prealloc_running_ = false;
  bool prealloc_needed_ = false;
  std::unique_ptr<std::thread> prealloc_thread_;
};

}  // namespace xllm

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

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace xllm {

PagedKVCachePageAllocator::PagedKVCachePageAllocator(
    KVCachePageBackend& backend)
    : backend_(backend) {}

PagedKVCachePageAllocator::~PagedKVCachePageAllocator() {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    prealloc_running_ = false;
    cond_.notify_all();
  }
  if (prealloc_thread_ && prealloc_thread_->joinable()) {
    prealloc_thread_->join();
  }
}

void PagedKVCachePageAllocator::init(size_t total_pages,
                                     size_t page_size,
                                     int32_t dp_size,
                                     bool enable_prealloc) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "PagedKVCachePageAllocator already initialized";
    return;
  }
  CHECK_GT(page_size, 0);
  CHECK_LE(page_size, static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  CHECK_GT(dp_size, 0);
  total_pages_ = total_pages;
  page_size_ = page_size;
  dp_size_ = dp_size;
  enable_prealloc_ = enable_prealloc;
  initialized_ = true;
}

bool PagedKVCachePageAllocator::is_initialized() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return initialized_;
}

bool PagedKVCachePageAllocator::register_model(const std::string& model_id,
                                               int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PagedKVCachePageAllocator not initialized";
  CHECK_GT(num_layers, 0);
  CHECK_LE(static_cast<uint64_t>(num_layers),
           std::numeric_limits<size_t>::max() / 2);
  auto [it, inserted] = model_states_.try_emplace(model_id);
  if (!inserted) {
    LOG(WARNING) << "Model " << model_id << " already registered";
    return false;
  }
  ModelState& state = it->second;
  state.num_layers = num_layers;
  state.phy_pages_per_virt_page = 2 * static_cast<size_t>(num_layers);
  state.num_total_virt_pages = total_pages_ / state.phy_pages_per_virt_page;
  CHECK_LE(state.num_total_virt_pages,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  state.dp_group_pages.resize(dp_size_);
  for (auto& pages : state.dp_group_pages) {
    pages.num_free_virt_pages = state.num_total_virt_pages;
    for (size_t index = 0; index < state.num_total_virt_pages; ++index) {
      pages.free_virt_page_list.emplace_back(static_cast<int64_t>(index));
    }
  }
  return true;
}

void PagedKVCachePageAllocator::start_prealloc_thread() {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PagedKVCachePageAllocator not initialized";
  if (!enable_prealloc_ || prealloc_thread_) {
    return;
  }
  prealloc_running_ = true;
  prealloc_needed_ = true;
  prealloc_thread_ = std::make_unique<std::thread>(
      &PagedKVCachePageAllocator::prealloc_worker, this);
  cond_.notify_all();
}

std::unique_ptr<KVCachePageState>
PagedKVCachePageAllocator::alloc_kv_cache_page(const std::string& model_id,
                                               int32_t dp_rank) {
  std::unique_lock<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PagedKVCachePageAllocator not initialized";
  validate_dp_rank(dp_rank);
  ModelState& state = get_model_state(model_id);
  wait_for_model_transition(state, lock);
  CHECK(!state.suspended) << "Cannot allocate from suspended model "
                          << model_id;
  auto& pages = state.dp_group_pages[dp_rank];
  if (!pages.reserved_virt_page_list.empty()) {
    int64_t page_id = pages.reserved_virt_page_list.front();
    pages.reserved_virt_page_list.pop_front();
    --pages.num_free_virt_pages;
    pages.allocated_virt_page_list.insert(page_id);
    if (enable_prealloc_ &&
        pages.reserved_virt_page_list.size() < kMinReservedPages) {
      prealloc_needed_ = true;
      cond_.notify_all();
    }
    return std::make_unique<KVCachePageState>(page_id, page_size_);
  }
  if (pages.free_virt_page_list.empty() ||
      !backend_.try_reserve_pages(
          model_id, dp_rank, state.phy_pages_per_virt_page)) {
    return nullptr;
  }
  int64_t page_id = pages.free_virt_page_list.front();
  pages.free_virt_page_list.pop_front();
  --pages.num_free_virt_pages;
  ++state.pending_map_ops;
  lock.unlock();
  map_virtual_pages(model_id, dp_rank, {page_id});
  lock.lock();
  pages.allocated_virt_page_list.insert(page_id);
  --state.pending_map_ops;
  cond_.notify_all();
  lock.unlock();
  trigger_preallocation();
  return std::make_unique<KVCachePageState>(page_id, page_size_);
}

void PagedKVCachePageAllocator::free_kv_cache_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virtual_page_ids) {
  std::unique_lock<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  ModelState& state = get_model_state(model_id);
  wait_for_model_transition(state, lock);
  auto& pages = state.dp_group_pages[dp_rank];
  for (int64_t page_id : virtual_page_ids) {
    CHECK_EQ(pages.allocated_virt_page_list.erase(page_id), 1)
        << "KV page " << page_id << " is not allocated for model " << model_id;
  }
  pages.num_free_virt_pages += virtual_page_ids.size();
  if (!state.kv_cache_mapped) {
    pages.free_virt_page_list.insert(pages.free_virt_page_list.end(),
                                     virtual_page_ids.begin(),
                                     virtual_page_ids.end());
    return;
  }
  size_t reserve_capacity =
      kMaxReservedPages -
      std::min(kMaxReservedPages, pages.reserved_virt_page_list.size());
  size_t num_to_reserve = std::min(reserve_capacity, virtual_page_ids.size());
  pages.reserved_virt_page_list.insert(
      pages.reserved_virt_page_list.end(),
      virtual_page_ids.begin(),
      virtual_page_ids.begin() + num_to_reserve);
  std::vector<int64_t> pages_to_unmap(virtual_page_ids.begin() + num_to_reserve,
                                      virtual_page_ids.end());
  if (pages_to_unmap.empty()) {
    return;
  }
  ++state.pending_map_ops;
  lock.unlock();
  unmap_virtual_pages(model_id, dp_rank, pages_to_unmap);
  lock.lock();
  pages.free_virt_page_list.insert(pages.free_virt_page_list.end(),
                                   pages_to_unmap.begin(),
                                   pages_to_unmap.end());
  backend_.release_pages(
      model_id, dp_rank, pages_to_unmap.size() * state.phy_pages_per_virt_page);
  --state.pending_map_ops;
  cond_.notify_all();
}

void PagedKVCachePageAllocator::trim_kv_cache(const std::string& model_id,
                                              int32_t dp_rank) {
  std::unique_lock<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  ModelState& state = get_model_state(model_id);
  wait_for_model_transition(state, lock);
  auto& pages = state.dp_group_pages[dp_rank];
  std::vector<int64_t> pages_to_unmap(pages.reserved_virt_page_list.begin(),
                                      pages.reserved_virt_page_list.end());
  pages.reserved_virt_page_list.clear();
  if (!state.kv_cache_mapped) {
    pages.free_virt_page_list.insert(pages.free_virt_page_list.end(),
                                     pages_to_unmap.begin(),
                                     pages_to_unmap.end());
    return;
  }
  if (pages_to_unmap.empty()) {
    return;
  }
  ++state.pending_map_ops;
  lock.unlock();
  unmap_virtual_pages(model_id, dp_rank, pages_to_unmap);
  lock.lock();
  pages.free_virt_page_list.insert(pages.free_virt_page_list.end(),
                                   pages_to_unmap.begin(),
                                   pages_to_unmap.end());
  backend_.release_pages(
      model_id, dp_rank, pages_to_unmap.size() * state.phy_pages_per_virt_page);
  --state.pending_map_ops;
  cond_.notify_all();
}

std::optional<KVCacheMappingPlan> PagedKVCachePageAllocator::begin_suspend(
    const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  ModelState& state = get_model_state(model_id);
  wait_for_model_transition(state, lock);
  if (state.suspended) {
    return std::nullopt;
  }
  state.suspended = true;
  state.transition = ModelTransition::SUSPENDING;
  cond_.notify_all();
  cond_.wait(lock, [&state] { return state.pending_map_ops == 0; });
  return snapshot_pages(state);
}

void PagedKVCachePageAllocator::finish_suspend(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  ModelState& state = get_model_state(model_id);
  CHECK(state.transition == ModelTransition::SUSPENDING);
  state.kv_cache_mapped = false;
  state.transition = ModelTransition::IDLE;
  cond_.notify_all();
}

std::optional<KVCacheMappingPlan> PagedKVCachePageAllocator::begin_resume(
    const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  ModelState& state = get_model_state(model_id);
  wait_for_model_transition(state, lock);
  if (!state.suspended) {
    return std::nullopt;
  }
  state.transition = ModelTransition::RESUMING;
  return snapshot_pages(state);
}

void PagedKVCachePageAllocator::finish_resume(const std::string& model_id,
                                              bool success) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    ModelState& state = get_model_state(model_id);
    CHECK(state.transition == ModelTransition::RESUMING);
    state.suspended = !success;
    state.kv_cache_mapped = success;
    state.transition = ModelTransition::IDLE;
    cond_.notify_all();
  }
  if (success) {
    trigger_preallocation();
  }
}

bool PagedKVCachePageAllocator::is_suspended(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  return it != model_states_.end() && it->second.suspended;
}

void PagedKVCachePageAllocator::map_pages(const std::string& model_id,
                                          const KVCacheMappingPlan& plan) {
  CHECK_EQ(plan.dp_group_page_ids.size(), static_cast<size_t>(dp_size_));
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    map_virtual_pages(model_id, dp_rank, plan.dp_group_page_ids[dp_rank]);
  }
}

void PagedKVCachePageAllocator::unmap_pages(const std::string& model_id,
                                            const KVCacheMappingPlan& plan) {
  CHECK_EQ(plan.dp_group_page_ids.size(), static_cast<size_t>(dp_size_));
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    unmap_virtual_pages(model_id, dp_rank, plan.dp_group_page_ids[dp_rank]);
  }
}

size_t PagedKVCachePageAllocator::get_num_free_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  return get_model_state(model_id).dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t PagedKVCachePageAllocator::get_num_inuse_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  const ModelState& state = get_model_state(model_id);
  return state.num_total_virt_pages -
         state.dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t PagedKVCachePageAllocator::get_num_reserved_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  return get_model_state(model_id)
      .dp_group_pages[dp_rank]
      .reserved_virt_page_list.size();
}

size_t PagedKVCachePageAllocator::get_num_total_virt_pages(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).num_total_virt_pages;
}

int64_t PagedKVCachePageAllocator::get_virt_page_id(
    int64_t block_id,
    size_t block_memory_size) const {
  CHECK_GE(block_id, 0);
  CHECK_GT(page_size_, 0);
  CHECK_GT(block_memory_size, 0);
  CHECK_LE(static_cast<uint64_t>(block_id),
           std::numeric_limits<uint64_t>::max() / block_memory_size);
  uint64_t page_id =
      static_cast<uint64_t>(block_id) * block_memory_size / page_size_;
  CHECK_LE(page_id, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  return static_cast<int64_t>(page_id);
}

int64_t PagedKVCachePageAllocator::get_offset(int64_t virtual_page_id) const {
  CHECK_GE(virtual_page_id, 0);
  CHECK_GT(page_size_, 0);
  CHECK_LE(
      static_cast<uint64_t>(virtual_page_id),
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / page_size_);
  return virtual_page_id * static_cast<int64_t>(page_size_);
}

int64_t PagedKVCachePageAllocator::num_layers(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).num_layers;
}

size_t PagedKVCachePageAllocator::phy_pages_per_virt_page(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).phy_pages_per_virt_page;
}

PagedKVCachePageAllocator::ModelState&
PagedKVCachePageAllocator::get_model_state(const std::string& model_id) {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

const PagedKVCachePageAllocator::ModelState&
PagedKVCachePageAllocator::get_model_state(const std::string& model_id) const {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

void PagedKVCachePageAllocator::wait_for_model_transition(
    ModelState& state,
    std::unique_lock<std::mutex>& lock) {
  cond_.wait(lock,
             [&state] { return state.transition == ModelTransition::IDLE; });
}

KVCacheMappingPlan PagedKVCachePageAllocator::snapshot_pages(
    const ModelState& state) const {
  KVCacheMappingPlan plan;
  plan.physical_pages_per_virtual_page = state.phy_pages_per_virt_page;
  plan.dp_group_page_ids.resize(dp_size_);
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    const auto& pages = state.dp_group_pages[dp_rank];
    auto& page_ids = plan.dp_group_page_ids[dp_rank];
    page_ids.reserve(pages.reserved_virt_page_list.size() +
                     pages.allocated_virt_page_list.size());
    page_ids.insert(page_ids.end(),
                    pages.reserved_virt_page_list.begin(),
                    pages.reserved_virt_page_list.end());
    page_ids.insert(page_ids.end(),
                    pages.allocated_virt_page_list.begin(),
                    pages.allocated_virt_page_list.end());
  }
  return plan;
}

void PagedKVCachePageAllocator::validate_dp_rank(int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0);
  CHECK_LT(dp_rank, dp_size_);
}

void PagedKVCachePageAllocator::map_virtual_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virtual_page_ids) {
  if (virtual_page_ids.empty()) {
    return;
  }
  std::vector<int64_t> offsets;
  offsets.reserve(virtual_page_ids.size());
  for (int64_t page_id : virtual_page_ids) {
    offsets.emplace_back(get_offset(page_id));
  }
  CHECK(backend_.map_pages(model_id, dp_rank, offsets))
      << "Failed to map KV pages for model=" << model_id
      << " dp_rank=" << dp_rank << "; mapping state is uncertain";
}

void PagedKVCachePageAllocator::unmap_virtual_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virtual_page_ids) {
  if (virtual_page_ids.empty()) {
    return;
  }
  std::vector<int64_t> offsets;
  offsets.reserve(virtual_page_ids.size());
  for (int64_t page_id : virtual_page_ids) {
    offsets.emplace_back(get_offset(page_id));
  }
  CHECK(backend_.unmap_pages(model_id, dp_rank, offsets))
      << "Failed to unmap KV pages for model=" << model_id
      << " dp_rank=" << dp_rank << "; mapping state is uncertain";
}

void PagedKVCachePageAllocator::trigger_preallocation() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!enable_prealloc_) {
    return;
  }
  prealloc_needed_ = true;
  cond_.notify_all();
}

void PagedKVCachePageAllocator::prealloc_worker() {
  while (true) {
    std::vector<PreallocationBatch> batches;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cond_.wait(lock,
                 [this] { return prealloc_needed_ || !prealloc_running_; });
      if (!prealloc_running_) {
        return;
      }
      prealloc_needed_ = false;
      batches.reserve(model_states_.size() * static_cast<size_t>(dp_size_));
      for (auto& [model_id, state] : model_states_) {
        if (state.suspended || state.transition != ModelTransition::IDLE) {
          continue;
        }
        for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
          auto& pages = state.dp_group_pages[dp_rank];
          size_t current_reserved = pages.reserved_virt_page_list.size();
          size_t to_reserve =
              kMinReservedPages - std::min(kMinReservedPages, current_reserved);
          to_reserve = std::min(to_reserve, pages.free_virt_page_list.size());
          to_reserve = std::min(to_reserve,
                                backend_.available_pages(model_id, dp_rank) /
                                    state.phy_pages_per_virt_page);
          if (to_reserve == 0 ||
              !backend_.try_reserve_pages(
                  model_id,
                  dp_rank,
                  to_reserve * state.phy_pages_per_virt_page)) {
            continue;
          }
          std::vector<int64_t> page_ids;
          page_ids.reserve(to_reserve);
          for (size_t index = 0; index < to_reserve; ++index) {
            page_ids.emplace_back(pages.free_virt_page_list.front());
            pages.free_virt_page_list.pop_front();
          }
          // Count batches at reservation time so suspend cannot miss an
          // unexecuted batch while it collects its mapping snapshot.
          ++state.pending_map_ops;
          batches.emplace_back(
              PreallocationBatch{model_id, dp_rank, std::move(page_ids)});
        }
      }
    }
    for (auto& batch : batches) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        ModelState& state = get_model_state(batch.model_id);
        if (state.suspended || !prealloc_running_) {
          auto& pages = state.dp_group_pages[batch.dp_rank];
          pages.free_virt_page_list.insert(pages.free_virt_page_list.begin(),
                                           batch.virtual_page_ids.begin(),
                                           batch.virtual_page_ids.end());
          backend_.release_pages(
              batch.model_id,
              batch.dp_rank,
              batch.virtual_page_ids.size() * state.phy_pages_per_virt_page);
          --state.pending_map_ops;
          cond_.notify_all();
          continue;
        }
      }
      map_virtual_pages(batch.model_id, batch.dp_rank, batch.virtual_page_ids);
      {
        std::lock_guard<std::mutex> lock(mtx_);
        ModelState& state = get_model_state(batch.model_id);
        auto& pages = state.dp_group_pages[batch.dp_rank];
        pages.reserved_virt_page_list.insert(
            pages.reserved_virt_page_list.end(),
            batch.virtual_page_ids.begin(),
            batch.virtual_page_ids.end());
        --state.pending_map_ops;
        cond_.notify_all();
      }
    }
  }
}

}  // namespace xllm

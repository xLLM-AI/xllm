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

#include <algorithm>
#include <limits>
#include <utility>

namespace xllm {

VmmKVPageAllocatorImpl::VmmKVPageAllocatorImpl(KVCachePageBackend& backend)
    : backend_(backend) {}

VmmKVPageAllocatorImpl::~VmmKVPageAllocatorImpl() { shutdown(); }

size_t VmmKVPageAllocatorImpl::DpGroupPages::num_unmapped_virt_pages(
    size_t num_total_virt_pages) const {
  return retry_virt_page_list.size() +
         (num_total_virt_pages - next_fresh_virt_page_id) +
         recycled_virt_page_list.size();
}

int64_t VmmKVPageAllocatorImpl::DpGroupPages::take_free_virt_page(
    size_t num_total_virt_pages) {
  if (!retry_virt_page_list.empty()) {
    int64_t page_id = retry_virt_page_list.front();
    retry_virt_page_list.pop_front();
    return page_id;
  }
  if (next_fresh_virt_page_id < num_total_virt_pages) {
    return static_cast<int64_t>(next_fresh_virt_page_id++);
  }
  CHECK(!recycled_virt_page_list.empty());
  int64_t page_id = recycled_virt_page_list.front();
  recycled_virt_page_list.pop_front();
  return page_id;
}

void VmmKVPageAllocatorImpl::DpGroupPages::restore_free_virt_pages(
    const std::vector<int64_t>& virtual_page_ids) {
  retry_virt_page_list.insert(retry_virt_page_list.begin(),
                              virtual_page_ids.begin(),
                              virtual_page_ids.end());
}

void VmmKVPageAllocatorImpl::DpGroupPages::recycle_free_virt_pages(
    const std::vector<int64_t>& virtual_page_ids) {
  recycled_virt_page_list.insert(recycled_virt_page_list.end(),
                                 virtual_page_ids.begin(),
                                 virtual_page_ids.end());
}

void VmmKVPageAllocatorImpl::shutdown() {
  std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
  {
    std::lock_guard<std::mutex> lock(mtx_);
    shutting_down_ = true;
    prealloc_running_ = false;
    cond_.notify_all();
  }
  if (prealloc_thread_ && prealloc_thread_->joinable()) {
    prealloc_thread_->join();
  }
  std::unique_lock<std::mutex> lock(mtx_);
  cond_.wait(lock, [this] {
    return std::all_of(
        model_states_.begin(), model_states_.end(), [](const auto& item) {
          return item.second.pending_map_ops == 0;
        });
  });
}

void VmmKVPageAllocatorImpl::init(size_t total_pages,
                                  size_t page_size,
                                  int32_t dp_size,
                                  bool enable_prealloc,
                                  size_t max_reserved_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "VmmKVPageAllocatorImpl already initialized";
    return;
  }
  CHECK_GT(page_size, 0);
  CHECK_LE(page_size, static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  CHECK_GT(dp_size, 0);
  CHECK_LE(max_reserved_pages, kMaxReservedPages);
  total_pages_ = total_pages;
  page_size_ = page_size;
  dp_size_ = dp_size;
  max_reserved_pages_ = max_reserved_pages;
  enable_prealloc_ = enable_prealloc && max_reserved_pages_ > 0;
  initialized_ = true;
}

bool VmmKVPageAllocatorImpl::is_initialized() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return initialized_;
}

bool VmmKVPageAllocatorImpl::register_model(
    const std::string& model_id,
    int64_t num_layers,
    size_t physical_pages_per_virtual_page,
    size_t max_virtual_pages) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "VmmKVPageAllocatorImpl not initialized";
  CHECK(!shutting_down_) << "VmmKVPageAllocatorImpl is shutting down";
  CHECK_GT(num_layers, 0);
  CHECK_LE(static_cast<uint64_t>(num_layers),
           std::numeric_limits<size_t>::max() / (2 * kMaxReservedPages));
  auto [it, inserted] = model_states_.try_emplace(model_id);
  if (!inserted) {
    LOG(WARNING) << "Model " << model_id << " already registered";
    return false;
  }
  ModelState& state = it->second;
  state.num_layers = num_layers;
  const size_t symmetric_pages = 2 * static_cast<size_t>(num_layers);
  state.phy_pages_per_virt_page = physical_pages_per_virtual_page == 0
                                      ? symmetric_pages
                                      : physical_pages_per_virtual_page;
  state.num_total_virt_pages = total_pages_ / state.phy_pages_per_virt_page;
  if (max_virtual_pages > 0) {
    state.num_total_virt_pages =
        std::min(state.num_total_virt_pages, max_virtual_pages);
  }
  // Preserve the symmetric allocator's physical preallocation budget when
  // one logical page expands into multiple hardware pages per tensor.
  state.min_reserved_pages =
      max_reserved_pages_ == 0
          ? 0
          : std::max<size_t>(1,
                             std::min(kMinReservedPages, max_reserved_pages_) *
                                 symmetric_pages /
                                 state.phy_pages_per_virt_page);
  state.max_reserved_pages =
      max_reserved_pages_ == 0
          ? 0
          : std::max<size_t>(1,
                             max_reserved_pages_ * symmetric_pages /
                                 state.phy_pages_per_virt_page);
  CHECK_LE(state.num_total_virt_pages,
           static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  state.dp_group_pages.resize(dp_size_);
  for (auto& pages : state.dp_group_pages) {
    pages.num_free_virt_pages = state.num_total_virt_pages;
  }
  return true;
}

bool VmmKVPageAllocatorImpl::unregister_model(const std::string& model_id) {
  std::unique_lock<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  if (it == model_states_.end()) {
    return false;
  }
  ModelState& state = it->second;
  const auto has_allocated_pages = [&state] {
    return std::any_of(state.dp_group_pages.begin(),
                       state.dp_group_pages.end(),
                       [](const DpGroupPages& pages) {
                         return !pages.allocated_virt_page_list.empty();
                       });
  };
  if (state.unregistering || state.transition != ModelTransition::IDLE ||
      state.transition_waiters > 0 || has_allocated_pages()) {
    LOG(ERROR) << "Cannot unregister active KV cache model " << model_id;
    return false;
  }
  state.unregistering = true;
  cond_.notify_all();
  cond_.wait(lock, [&state] { return state.pending_map_ops == 0; });
  // A synchronous allocation already mapping when cleanup began can finish
  // with a live page. Keep its state available for the owner to recycle it.
  if (has_allocated_pages()) {
    state.unregistering = false;
    cond_.notify_all();
    return false;
  }
  const bool mapped = state.kv_cache_mapped;
  const auto plan = snapshot_pages(state);
  lock.unlock();
  if (mapped) {
    unmap_pages(model_id, plan);
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      backend_.release_pages(model_id,
                             dp_rank,
                             plan.dp_group_page_ids[dp_rank].size() *
                                 plan.physical_pages_per_virtual_page);
    }
  }
  lock.lock();
  model_states_.erase(model_id);
  return true;
}

void VmmKVPageAllocatorImpl::start_prealloc_thread(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "VmmKVPageAllocatorImpl not initialized";
  CHECK(!shutting_down_) << "VmmKVPageAllocatorImpl is shutting down";
  ModelState& state = get_model_state(model_id);
  CHECK(!state.unregistering)
      << "Cannot preallocate for unregistering model " << model_id;
  state.ready_for_preallocation = true;
  if (!enable_prealloc_) {
    return;
  }
  prealloc_needed_ = true;
  if (!prealloc_thread_) {
    prealloc_running_ = true;
    prealloc_thread_ = std::make_unique<std::thread>(
        &VmmKVPageAllocatorImpl::prealloc_worker, this);
  }
  cond_.notify_all();
}

std::unique_ptr<VirtualKVCachePageState>
VmmKVPageAllocatorImpl::alloc_kv_cache_page(const std::string& model_id,
                                            int32_t dp_rank) {
  std::unique_lock<std::mutex> lock(mtx_);
  CHECK(initialized_) << "VmmKVPageAllocatorImpl not initialized";
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
        pages.reserved_virt_page_list.size() < state.min_reserved_pages) {
      prealloc_needed_ = true;
      cond_.notify_all();
    }
    return std::make_unique<VirtualKVCachePageState>(page_id, page_size_);
  }
  const size_t unmapped_virt_pages =
      pages.num_unmapped_virt_pages(state.num_total_virt_pages);
  if (unmapped_virt_pages == 0) {
    LOG(WARNING) << "VMM KV page allocation failed: "
                    "reason=virtual_capacity_exhausted"
                 << " model_id=" << model_id << " dp_rank=" << dp_rank
                 << " total_virtual_pages=" << state.num_total_virt_pages
                 << " free_virtual_pages=" << pages.num_free_virt_pages
                 << " reserved_virtual_pages="
                 << pages.reserved_virt_page_list.size()
                 << " allocated_virtual_pages="
                 << pages.allocated_virt_page_list.size()
                 << " unmapped_virtual_pages=" << unmapped_virt_pages;
    return nullptr;
  }
  if (!backend_.try_reserve_pages(
          model_id, dp_rank, state.phy_pages_per_virt_page)) {
    LOG(WARNING) << "VMM KV page allocation failed: "
                    "reason=physical_page_budget_rejected"
                 << " model_id=" << model_id << " dp_rank=" << dp_rank
                 << " requested_physical_pages="
                 << state.phy_pages_per_virt_page
                 << " total_virtual_pages=" << state.num_total_virt_pages
                 << " free_virtual_pages=" << pages.num_free_virt_pages
                 << " reserved_virtual_pages="
                 << pages.reserved_virt_page_list.size()
                 << " allocated_virtual_pages="
                 << pages.allocated_virt_page_list.size()
                 << " unmapped_virtual_pages=" << unmapped_virt_pages;
    return nullptr;
  }
  int64_t page_id = pages.take_free_virt_page(state.num_total_virt_pages);
  --pages.num_free_virt_pages;
  ++state.pending_map_ops;
  lock.unlock();
  const bool mapped = map_virtual_pages(
      model_id, dp_rank, {page_id}, MappingSource::ALLOCATION);
  lock.lock();
  if (!mapped) {
    // The backend confirms rollback on every participating worker before the
    // virtual ID and shared budget can become available to another request.
    pages.restore_free_virt_pages({page_id});
    ++pages.num_free_virt_pages;
    backend_.release_pages(model_id, dp_rank, state.phy_pages_per_virt_page);
    --state.pending_map_ops;
    cond_.notify_all();
    return nullptr;
  }
  pages.allocated_virt_page_list.insert(page_id);
  --state.pending_map_ops;
  cond_.notify_all();
  lock.unlock();
  trigger_preallocation();
  return std::make_unique<VirtualKVCachePageState>(page_id, page_size_);
}

void VmmKVPageAllocatorImpl::free_kv_cache_pages(
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
    pages.recycle_free_virt_pages(virtual_page_ids);
    return;
  }
  size_t reserve_capacity =
      state.max_reserved_pages -
      std::min(state.max_reserved_pages, pages.reserved_virt_page_list.size());
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
  pages.recycle_free_virt_pages(pages_to_unmap);
  backend_.release_pages(
      model_id, dp_rank, pages_to_unmap.size() * state.phy_pages_per_virt_page);
  --state.pending_map_ops;
  cond_.notify_all();
}

void VmmKVPageAllocatorImpl::trim_kv_cache(const std::string& model_id,
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
    pages.recycle_free_virt_pages(pages_to_unmap);
    return;
  }
  if (pages_to_unmap.empty()) {
    return;
  }
  ++state.pending_map_ops;
  lock.unlock();
  unmap_virtual_pages(model_id, dp_rank, pages_to_unmap);
  lock.lock();
  pages.recycle_free_virt_pages(pages_to_unmap);
  backend_.release_pages(
      model_id, dp_rank, pages_to_unmap.size() * state.phy_pages_per_virt_page);
  --state.pending_map_ops;
  cond_.notify_all();
}

std::optional<KVCacheMappingPlan> VmmKVPageAllocatorImpl::begin_suspend(
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

void VmmKVPageAllocatorImpl::finish_suspend(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  ModelState& state = get_model_state(model_id);
  CHECK(state.transition == ModelTransition::SUSPENDING);
  state.kv_cache_mapped = false;
  state.transition = ModelTransition::IDLE;
  cond_.notify_all();
}

std::optional<KVCacheMappingPlan> VmmKVPageAllocatorImpl::begin_resume(
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

void VmmKVPageAllocatorImpl::finish_resume(const std::string& model_id,
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

bool VmmKVPageAllocatorImpl::is_suspended(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_states_.find(model_id);
  return it != model_states_.end() && it->second.suspended;
}

bool VmmKVPageAllocatorImpl::map_pages(const std::string& model_id,
                                       const KVCacheMappingPlan& plan) {
  CHECK_EQ(plan.dp_group_page_ids.size(), static_cast<size_t>(dp_size_));
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    if (map_virtual_pages(model_id,
                          dp_rank,
                          plan.dp_group_page_ids[dp_rank],
                          MappingSource::MAPPING_PLAN)) {
      continue;
    }
    // This group rolled itself back. Revert earlier groups before the caller
    // restores the combined lifecycle reservation.
    for (int32_t mapped_rank = 0; mapped_rank < dp_rank; ++mapped_rank) {
      unmap_virtual_pages(
          model_id, mapped_rank, plan.dp_group_page_ids[mapped_rank]);
    }
    return false;
  }
  return true;
}

void VmmKVPageAllocatorImpl::unmap_pages(const std::string& model_id,
                                         const KVCacheMappingPlan& plan) {
  CHECK_EQ(plan.dp_group_page_ids.size(), static_cast<size_t>(dp_size_));
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    unmap_virtual_pages(model_id, dp_rank, plan.dp_group_page_ids[dp_rank]);
  }
}

size_t VmmKVPageAllocatorImpl::get_num_available_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  const ModelState& state = get_model_state(model_id);
  if (shutting_down_ || state.unregistering || state.suspended ||
      state.transition != ModelTransition::IDLE) {
    return 0;
  }
  const auto& pages = state.dp_group_pages[dp_rank];
  const size_t growth =
      std::min(pages.num_unmapped_virt_pages(state.num_total_virt_pages),
               backend_.available_pages(model_id, dp_rank) /
                   state.phy_pages_per_virt_page);
  return pages.reserved_virt_page_list.size() + growth;
}

size_t VmmKVPageAllocatorImpl::get_num_free_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  return get_model_state(model_id).dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t VmmKVPageAllocatorImpl::get_num_inuse_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  const ModelState& state = get_model_state(model_id);
  return state.num_total_virt_pages -
         state.dp_group_pages[dp_rank].num_free_virt_pages;
}

size_t VmmKVPageAllocatorImpl::get_num_reserved_virt_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mtx_);
  validate_dp_rank(dp_rank);
  return get_model_state(model_id)
      .dp_group_pages[dp_rank]
      .reserved_virt_page_list.size();
}

size_t VmmKVPageAllocatorImpl::get_num_total_virt_pages(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).num_total_virt_pages;
}

int64_t VmmKVPageAllocatorImpl::get_virt_page_id(
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

int64_t VmmKVPageAllocatorImpl::get_offset(int64_t virtual_page_id) const {
  CHECK_GE(virtual_page_id, 0);
  CHECK_GT(page_size_, 0);
  CHECK_LE(
      static_cast<uint64_t>(virtual_page_id),
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / page_size_);
  return virtual_page_id * static_cast<int64_t>(page_size_);
}

int64_t VmmKVPageAllocatorImpl::num_layers(const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).num_layers;
}

size_t VmmKVPageAllocatorImpl::phy_pages_per_virt_page(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  return get_model_state(model_id).phy_pages_per_virt_page;
}

VmmKVPageAllocatorImpl::ModelState& VmmKVPageAllocatorImpl::get_model_state(
    const std::string& model_id) {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

const VmmKVPageAllocatorImpl::ModelState&
VmmKVPageAllocatorImpl::get_model_state(const std::string& model_id) const {
  auto it = model_states_.find(model_id);
  CHECK(it != model_states_.end()) << "Model " << model_id << " not registered";
  return it->second;
}

void VmmKVPageAllocatorImpl::wait_for_model_transition(
    ModelState& state,
    std::unique_lock<std::mutex>& lock) {
  CHECK(!shutting_down_) << "VmmKVPageAllocatorImpl is shutting down";
  CHECK(!state.unregistering) << "KV cache model is unregistering";
  ++state.transition_waiters;
  cond_.wait(lock, [this, &state] {
    return shutting_down_ || state.transition == ModelTransition::IDLE;
  });
  --state.transition_waiters;
  CHECK(!shutting_down_) << "VmmKVPageAllocatorImpl is shutting down";
  CHECK(!state.unregistering) << "KV cache model is unregistering";
}

KVCacheMappingPlan VmmKVPageAllocatorImpl::snapshot_pages(
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

void VmmKVPageAllocatorImpl::validate_dp_rank(int32_t dp_rank) const {
  CHECK_GE(dp_rank, 0);
  CHECK_LT(dp_rank, dp_size_);
}

bool VmmKVPageAllocatorImpl::map_virtual_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& virtual_page_ids,
    MappingSource source) {
  if (virtual_page_ids.empty()) {
    return true;
  }
  std::vector<int64_t> offsets;
  offsets.reserve(virtual_page_ids.size());
  for (int64_t page_id : virtual_page_ids) {
    offsets.emplace_back(get_offset(page_id));
  }
  const bool mapped = backend_.map_pages(model_id, dp_rank, offsets);
  if (!mapped) {
    LOG(WARNING) << "VMM KV page mapping failed: "
                    "reason=backend_mapping_failure"
                 << " source=" << mapping_source_name(source)
                 << " model_id=" << model_id << " dp_rank=" << dp_rank
                 << " virtual_page_count=" << virtual_page_ids.size()
                 << " offset_count=" << offsets.size();
  }
  return mapped;
}

const char* VmmKVPageAllocatorImpl::mapping_source_name(MappingSource source) {
  switch (source) {
    case MappingSource::ALLOCATION:
      return "allocation";
    case MappingSource::MAPPING_PLAN:
      return "mapping_plan";
    case MappingSource::PREALLOCATION:
      return "preallocation";
  }
  return "unknown";
}

void VmmKVPageAllocatorImpl::unmap_virtual_pages(
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

void VmmKVPageAllocatorImpl::trigger_preallocation() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!enable_prealloc_ || shutting_down_) {
    return;
  }
  prealloc_needed_ = true;
  cond_.notify_all();
}

void VmmKVPageAllocatorImpl::prealloc_worker() {
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
        if (!state.ready_for_preallocation || state.unregistering ||
            state.suspended || state.transition != ModelTransition::IDLE) {
          continue;
        }
        for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
          auto& pages = state.dp_group_pages[dp_rank];
          size_t current_reserved = pages.reserved_virt_page_list.size();
          size_t to_reserve =
              state.min_reserved_pages -
              std::min(state.min_reserved_pages, current_reserved);
          to_reserve = std::min(
              to_reserve,
              pages.num_unmapped_virt_pages(state.num_total_virt_pages));
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
            page_ids.emplace_back(
                pages.take_free_virt_page(state.num_total_virt_pages));
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
        if (state.unregistering || state.suspended || !prealloc_running_) {
          auto& pages = state.dp_group_pages[batch.dp_rank];
          pages.restore_free_virt_pages(batch.virtual_page_ids);
          backend_.release_pages(
              batch.model_id,
              batch.dp_rank,
              batch.virtual_page_ids.size() * state.phy_pages_per_virt_page);
          --state.pending_map_ops;
          cond_.notify_all();
          continue;
        }
      }
      const bool mapped = map_virtual_pages(batch.model_id,
                                            batch.dp_rank,
                                            batch.virtual_page_ids,
                                            MappingSource::PREALLOCATION);
      {
        std::lock_guard<std::mutex> lock(mtx_);
        ModelState& state = get_model_state(batch.model_id);
        auto& pages = state.dp_group_pages[batch.dp_rank];
        if (mapped) {
          pages.reserved_virt_page_list.insert(
              pages.reserved_virt_page_list.end(),
              batch.virtual_page_ids.begin(),
              batch.virtual_page_ids.end());
        } else {
          pages.restore_free_virt_pages(batch.virtual_page_ids);
          backend_.release_pages(
              batch.model_id,
              batch.dp_rank,
              batch.virtual_page_ids.size() * state.phy_pages_per_virt_page);
        }
        --state.pending_map_ops;
        cond_.notify_all();
      }
    }
  }
}

}  // namespace xllm

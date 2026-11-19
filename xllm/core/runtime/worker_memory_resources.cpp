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

#include "core/runtime/worker_memory_resources.h"

#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <unordered_set>

#include "core/framework/allocator/global_virtual_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/platform/device.h"
#include "core/platform/vmm_api.h"

namespace xllm {

WorkerMemoryResources::WorkerMemoryResources()
    : weight_memory_(PhysicalPagePool::get_instance(),
                     GlobalVirtualMemoryRegion::get_instance()) {}

WorkerMemoryResources::~WorkerMemoryResources() {
  if (initialized_) {
    destroy();
  }
}

void WorkerMemoryResources::destroy() {
  std::lock_guard<std::mutex> lock(mutex_);
  mapping_transaction_.reset();
  retired_mapping_transaction_id_ = 0;
  weight_transaction_.reset();
  retired_weight_transaction_id_ = 0;
  execution_memory_reserve_bytes_ = 0;
  growth_memory_reserve_bytes_ = 0;
  weight_memory_.clear();
  kv_cache_memory_.clear();
  GlobalVirtualMemoryRegion::get_instance().reset();
  CHECK(PhysicalPagePool::get_instance().reset())
      << "Cannot reset physical page pool while pages are in use";
  pool_capacity_pages_ = 0;
  initialized_ = false;
  pool_ready_cv_.notify_all();
}

void WorkerMemoryResources::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    LOG(WARNING)
        << "WorkerMemoryResources already initialized, ignoring re-init";
    return;
  }

  device_ = device;
  init_device();
  weight_memory_.init(device_);
  kv_cache_memory_.init(
      device_, KVCacheConfig::get_instance().phy_page_granularity_size());
  initialized_ = true;
}

void WorkerMemoryResources::init_physical_page_pool(int64_t num_pages,
                                                    bool lazy) {
  init_physical_page_pool(
      num_pages, num_pages, lazy, /*growth_memory_reserve_bytes=*/0);
}

void WorkerMemoryResources::init_physical_page_pool(
    int64_t max_pages,
    int64_t initial_pages,
    bool lazy,
    size_t growth_memory_reserve_bytes) {
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(initialized_) << "Worker memory resources must be initialized first";
  const size_t page_size = static_cast<size_t>(
      KVCacheConfig::get_instance().phy_page_granularity_size());
  CHECK_GT(max_pages, 0);
  CHECK_GE(initial_pages, 0);
  CHECK_LE(initial_pages, max_pages);
  PhysicalPagePool::get_instance().init(device_,
                                        static_cast<size_t>(max_pages),
                                        static_cast<size_t>(initial_pages),
                                        page_size,
                                        lazy);
  growth_memory_reserve_bytes_ = growth_memory_reserve_bytes;
  PhysicalPagePool::get_instance().set_growth_reserve(
      growth_memory_reserve_bytes_);
  const int64_t configured_reserve =
      KVCacheConfig::get_instance().vmm_memory_reserve_bytes();
  CHECK_GE(configured_reserve, 0);
  execution_memory_reserve_bytes_ = static_cast<size_t>(configured_reserve);
  PhysicalPagePool::get_instance().set_allocation_headroom(
      execution_memory_reserve_bytes_);
  GlobalVirtualMemoryRegion::get_instance().init(device_);
  // Publish only after every local resource is ready for model startup.
  pool_capacity_pages_ = max_pages;
  lock.unlock();
  pool_ready_cv_.notify_all();
}

int64_t WorkerMemoryResources::wait_for_physical_page_pool(
    std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!pool_ready_cv_.wait_for(lock, timeout, [this]() {
        return pool_capacity_pages_ > 0 || !initialized_;
      })) {
    return 0;
  }
  return pool_capacity_pages_;
}

WorkerMemoryCapacity WorkerMemoryResources::memory_capacity_snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  WorkerMemoryCapacity snapshot;
  if (!initialized_) {
    return snapshot;
  }
  Device device(device_);
  device.set_device();
  snapshot.available_memory = device.free_memory();
  snapshot.total_memory = device.total_memory();
  snapshot.execution_memory_reserve_bytes = execution_memory_reserve_bytes_;
  snapshot.growth_memory_reserve_bytes = growth_memory_reserve_bytes_;
  const auto& pool = PhysicalPagePool::get_instance();
  snapshot.pool_initialized = pool.is_initialized();
  if (snapshot.pool_initialized) {
    snapshot.pool_capacity_pages = pool.num_total();
    snapshot.pool_committed_pages = pool.num_committed();
    snapshot.pool_allocatable_pages = pool.num_allocatable();
    snapshot.pool_active_pages = pool.num_active();
  }
  return snapshot;
}

void WorkerMemoryResources::set_execution_memory_reserve(size_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Models share one pool. A smaller profile must not lower another model's
  // established execution headroom.
  execution_memory_reserve_bytes_ =
      std::max(execution_memory_reserve_bytes_, bytes);
  PhysicalPagePool::get_instance().set_allocation_headroom(
      execution_memory_reserve_bytes_);
  // Return idle physical pages before execution needs the profiled headroom.
  if (KVCacheConfig::get_instance().vmm_elastic()) {
    PhysicalPagePool::get_instance().trim_idle(/*keep_pages=*/0);
  }
}

size_t WorkerMemoryResources::trim_idle_physical_pages(size_t keep_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  return PhysicalPagePool::get_instance().trim_idle(keep_pages);
}

bool WorkerMemoryResources::prepare_kv_mapping(
    uint64_t transaction_id,
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (transaction_id == 0) {
    return false;
  }
  if (mapping_transaction_.has_value()) {
    const KvMappingTransaction& current = *mapping_transaction_;
    if (transaction_id == current.id) {
      return current.model_id == model_id && current.offsets == offsets;
    }
    if (transaction_id < current.id || !current.committed) {
      return false;
    }
    // A newer prepare transfers the old committed mappings to allocator
    // ownership. Only one transaction's undo metadata remains resident.
    retired_mapping_transaction_id_ =
        std::max(retired_mapping_transaction_id_, current.id);
    mapping_transaction_.reset();
  }
  if (transaction_id <= retired_mapping_transaction_id_) {
    return false;
  }
  // Existing mappings are rejected before mutation. Abort must never remove
  // pages belonging to another allocation; a failed local map is atomic.
  if (!kv_cache_memory_.can_prepare_mapping(model_id, offsets) ||
      !kv_cache_memory_.map(model_id, offsets)) {
    retired_mapping_transaction_id_ = transaction_id;
    return false;
  }
  mapping_transaction_ =
      KvMappingTransaction{transaction_id, model_id, offsets, false};
  return true;
}

bool WorkerMemoryResources::commit_kv_mapping(uint64_t transaction_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!mapping_transaction_.has_value() ||
      mapping_transaction_->id != transaction_id) {
    return false;
  }
  // Retain undo state after commit because another worker may reject commit
  // or lose its reply, requiring rollback of the complete worker group.
  mapping_transaction_->committed = true;
  return true;
}

bool WorkerMemoryResources::abort_kv_mapping(uint64_t transaction_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (transaction_id == 0) {
    return false;
  }
  if (mapping_transaction_.has_value() &&
      mapping_transaction_->id == transaction_id) {
    if (!kv_cache_memory_.unmap(mapping_transaction_->model_id,
                                mapping_transaction_->offsets)) {
      return false;
    }
    mapping_transaction_.reset();
    if (KVCacheConfig::get_instance().vmm_elastic()) {
      PhysicalPagePool::get_instance().trim_idle(/*keep_pages=*/0);
    }
  } else if (mapping_transaction_.has_value() &&
             mapping_transaction_->id < transaction_id) {
    if (!mapping_transaction_->committed) {
      return false;
    }
    // A newer abort may arrive before its prepare. Retire an older committed
    // record without unmapping its allocator-owned pages.
    mapping_transaction_.reset();
  }
  // Record unseen aborts as well. A prepare dispatched earlier may reach the
  // worker later, and must observe this tombstone under the same mutex.
  retired_mapping_transaction_id_ =
      std::max(retired_mapping_transaction_id_, transaction_id);
  return true;
}

std::optional<int64_t> WorkerMemoryResources::get_kv_cache_num_layers(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return kv_cache_memory_.num_layers(model_id);
}

std::optional<WeightAllocationInfo>
WorkerMemoryResources::get_weight_allocation_info(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.get_weight_allocation_info(model_id);
}

std::vector<torch::Tensor> WorkerMemoryResources::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mutex_);
  return kv_cache_memory_.create_k_tensors(model_id, dims, dtype, num_layers);
}

std::vector<torch::Tensor> WorkerMemoryResources::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::ScalarType dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mutex_);
  return kv_cache_memory_.create_v_tensors(model_id, dims, dtype, num_layers);
}

bool WorkerMemoryResources::map_to_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  return kv_cache_memory_.map(model_id, offsets);
}

bool WorkerMemoryResources::unmap_from_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  const bool success = kv_cache_memory_.unmap(model_id, offsets);
  if (success && mapping_transaction_.has_value() &&
      mapping_transaction_->model_id == model_id) {
    // Normal lifecycle release ends transaction ownership of these offsets.
    // A delayed abort must not unmap pages subsequently reused at the same VA.
    const std::unordered_set<offset_t> released(offsets.begin(), offsets.end());
    auto& owned = mapping_transaction_->offsets;
    owned.erase(std::remove_if(owned.begin(),
                               owned.end(),
                               [&released](offset_t offset) {
                                 return released.find(offset) != released.end();
                               }),
                owned.end());
    if (owned.empty()) {
      retired_mapping_transaction_id_ =
          std::max(retired_mapping_transaction_id_, mapping_transaction_->id);
      mapping_transaction_.reset();
    }
  }
  if (success && KVCacheConfig::get_instance().vmm_elastic()) {
    PhysicalPagePool::get_instance().trim_idle(/*keep_pages=*/0);
  }
  return success;
}

bool WorkerMemoryResources::alloc_weight_pages(const std::string& model_id,
                                               size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.alloc_weight_pages(model_id, num_pages);
}

bool WorkerMemoryResources::prepare_weight_pages(uint64_t transaction_id,
                                                 const std::string& model_id,
                                                 size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (transaction_id == 0 || num_pages == 0) {
    return false;
  }
  if (weight_transaction_.has_value()) {
    const WeightMappingTransaction& current = *weight_transaction_;
    if (current.id == transaction_id) {
      return current.model_id == model_id && current.num_pages == num_pages;
    }
    if (transaction_id < current.id || !current.committed) {
      return false;
    }
    retired_weight_transaction_id_ =
        std::max(retired_weight_transaction_id_, current.id);
    weight_transaction_.reset();
  }
  if (transaction_id <= retired_weight_transaction_id_) {
    return false;
  }
  // The weight manager rejects a model that already owns pages before making
  // changes. Only successful fresh allocations belong to this transaction.
  if (!weight_memory_.alloc_weight_pages(model_id, num_pages)) {
    retired_weight_transaction_id_ = transaction_id;
    return false;
  }
  weight_transaction_ =
      WeightMappingTransaction{transaction_id, model_id, num_pages, false};
  return true;
}

bool WorkerMemoryResources::commit_weight_pages(uint64_t transaction_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!weight_transaction_.has_value() ||
      weight_transaction_->id != transaction_id) {
    return false;
  }
  weight_transaction_->committed = true;
  return true;
}

bool WorkerMemoryResources::abort_weight_pages(uint64_t transaction_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (transaction_id == 0) {
    return false;
  }
  if (weight_transaction_.has_value() &&
      weight_transaction_->id == transaction_id) {
    // Never free by model name unless this transaction created that model's
    // pages. A rejected prepare may have encountered an existing allocation.
    if (weight_memory_.free_weight(weight_transaction_->model_id) !=
        weight_transaction_->num_pages) {
      return false;
    }
    weight_transaction_.reset();
    if (KVCacheConfig::get_instance().vmm_elastic()) {
      PhysicalPagePool::get_instance().trim_idle(/*keep_pages=*/0);
    }
  } else if (weight_transaction_.has_value() &&
             weight_transaction_->id < transaction_id) {
    if (!weight_transaction_->committed) {
      return false;
    }
    weight_transaction_.reset();
  }
  retired_weight_transaction_id_ =
      std::max(retired_weight_transaction_id_, transaction_id);
  return true;
}

bool WorkerMemoryResources::allocate_weight(const std::string& model_id,
                                            void*& ptr,
                                            size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.allocate_weight(model_id, ptr, size);
}

void WorkerMemoryResources::init_device() {
  Device device(device_);
  device.set_device();
  device.init_device_context();

  const size_t chunk_sz = vmm::get_recommended_granularity(device_.index());
  KVCacheConfig::get_instance().phy_page_granularity_size(chunk_sz);
  LOG(INFO) << "Device initialized with granularity size: " << chunk_sz
            << " bytes";
}

size_t WorkerMemoryResources::free_weight(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t freed = weight_memory_.free_weight(model_id);
  if (weight_transaction_.has_value() &&
      weight_transaction_->model_id == model_id) {
    // The model name can be reused after release. Retire its old undo record
    // before a later allocation can become visible under that same name.
    retired_weight_transaction_id_ =
        std::max(retired_weight_transaction_id_, weight_transaction_->id);
    weight_transaction_.reset();
  }
  if (freed > 0 && KVCacheConfig::get_instance().vmm_elastic()) {
    PhysicalPagePool::get_instance().trim_idle(/*keep_pages=*/0);
  }
  return freed;
}

std::optional<VirtualKVBlockGlobalOffsets>
WorkerMemoryResources::get_global_offsets_for_block(const std::string& model_id,
                                                    int64_t layer_id,
                                                    int64_t block_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!GlobalVirtualMemoryRegion::get_instance().is_initialized()) {
    LOG(ERROR) << "GlobalVirtualMemoryRegion not initialized";
    return std::nullopt;
  }
  return kv_cache_memory_.get_global_offsets_for_block(
      model_id, layer_id, block_id);
}

std::pair<uint64_t, uint64_t>
WorkerMemoryResources::get_global_offsets_for_block(const std::string& model_id,
                                                    int64_t layer_id,
                                                    int64_t block_id,
                                                    size_t block_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!GlobalVirtualMemoryRegion::get_instance().is_initialized()) {
    LOG(ERROR) << "GlobalVirtualMemoryRegion not initialized";
    return {UINT64_MAX, UINT64_MAX};
  }
  return kv_cache_memory_.get_global_offsets_for_block(
      model_id, layer_id, block_id, block_size);
}

std::vector<WeightSegment> WorkerMemoryResources::get_model_weight_segments(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.get_model_weight_segments(model_id);
}

std::unordered_map<std::string, std::vector<WeightSegment>>
WorkerMemoryResources::get_all_model_weight_segments() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.get_all_model_weight_segments();
}

}  // namespace xllm

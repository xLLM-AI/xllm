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

#include <mutex>

#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/platform/device.h"
#include "core/platform/vmm_api.h"

namespace xllm {

WorkerMemoryResources::WorkerMemoryResources()
    : weight_memory_(PhysicalPagePool::get_instance(),
                     GlobalMemoryRegion::get_instance()) {}

WorkerMemoryResources::~WorkerMemoryResources() {
  if (initialized_) {
    destroy();
  }
}

void WorkerMemoryResources::destroy() {
  std::lock_guard<std::mutex> lock(mutex_);
  weight_memory_.clear();
  kv_cache_memory_.clear();
  GlobalMemoryRegion::get_instance().reset();
  CHECK(PhysicalPagePool::get_instance().reset())
      << "Cannot reset physical page pool while pages are in use";
  initialized_ = false;
}

void WorkerMemoryResources::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    LOG(WARNING)
        << "WorkerMemoryResources already initialized, ignoring re-init";
    return;
  }

  device_ = device;
  init_device_();
  weight_memory_.init(device_);
  kv_cache_memory_.init(
      device_, KVCacheConfig::get_instance().phy_page_granularity_size());
  initialized_ = true;
}

void WorkerMemoryResources::init_physical_page_pool(int64_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "Worker memory resources must be initialized first";
  const size_t page_size = static_cast<size_t>(
      KVCacheConfig::get_instance().phy_page_granularity_size());
  PhysicalPagePool::get_instance().init(
      device_, static_cast<size_t>(num_pages), page_size);
  GlobalMemoryRegion::get_instance().init(device_);
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
    torch::Dtype dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mutex_);
  return kv_cache_memory_.create_k_tensors(model_id, dims, dtype, num_layers);
}

std::vector<torch::Tensor> WorkerMemoryResources::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
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
  return kv_cache_memory_.unmap(model_id, offsets);
}

bool WorkerMemoryResources::alloc_weight_pages(const std::string& model_id,
                                               size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.alloc_weight_pages(model_id, num_pages);
}

bool WorkerMemoryResources::allocate_weight(const std::string& model_id,
                                            void*& ptr,
                                            size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  return weight_memory_.allocate_weight(model_id, ptr, size);
}

void WorkerMemoryResources::init_device_() {
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
  return weight_memory_.free_weight(model_id);
}

std::pair<uint64_t, uint64_t>
WorkerMemoryResources::get_global_offsets_for_block(const std::string& model_id,
                                                    int64_t layer_id,
                                                    int64_t block_id,
                                                    size_t block_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!GlobalMemoryRegion::get_instance().is_initialized()) {
    LOG(ERROR) << "GlobalMemoryRegion not initialized";
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

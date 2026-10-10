/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/allocator/model_memory_manager.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/distributed_runtime/collective_service.h"
#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/virtual_memory/mapped_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/weight/weight_transfer_segments.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/platform/device.h"
#include "core/platform/vmm_api.h"
#include "server/xllm_server_registry.h"

namespace xllm {

ModelMemoryManager::~ModelMemoryManager() {
  if (!initialized_) {
    return;
  }

  // Stop collective server if running
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(
          cluster_.collective_server_name());
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(
        cluster_.collective_server_name());
  }

  destroy();
}

void ModelMemoryManager::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);

  // Release model-owned mappings before tearing down the global address space.
  for (const auto& [model_id, weight] : weight_store_.models()) {
    if (weight.num_pages() == 0) {
      continue;
    }
    if (weight.is_fragmented()) {
      continue;
    }

    std::vector<page_id_t> page_ids;
    page_ids.reserve(weight.num_pages());
    for (size_t i = 0; i < weight.num_pages(); ++i) {
      page_ids.push_back(weight.start_page_id() + static_cast<page_id_t>(i));
    }
    PhysicalPagePool::get_instance().release_reserved_pages(page_ids);
    VLOG(1) << "Released weight pages for model " << model_id;
  }
  weight_store_.clear();
  kv_cache_regions_.clear();
  model_parallel_strategies_.clear();
  cluster_.clear();
  GlobalMemoryRegion::get_instance().reset();
  CHECK(PhysicalPagePool::get_instance().reset())
      << "Cannot reset physical page pool while pages are in use";
  initialized_ = false;
}

void ModelMemoryManager::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "ModelMemoryManager already initialized, ignoring re-init";
    return;
  }

  dev_ = device;
  init_device_();
  initialized_ = true;
}

std::optional<int64_t> ModelMemoryManager::get_kv_cache_num_layers(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const auto it = kv_cache_regions_.find(model_id);
  if (it == kv_cache_regions_.end()) {
    return std::nullopt;
  }
  return it->second.num_layers();
}

std::optional<WeightAllocationInfo>
ModelMemoryManager::get_weight_allocation_info(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const WeightAllocation* weight = weight_store_.find(model_id);
  if (weight == nullptr) {
    return std::nullopt;
  }
  return WeightAllocationInfo{weight->base_ptr(), weight->num_pages()};
}

// ============== Multi-node Setup ==============

void ModelMemoryManager::setup_multi_node_model_memory_dist(
    const ModelMemoryOptions& options,
    const std::string& master_node_addr,
    int32_t dp_size) {
  const auto& devices = options.devices();
  // The allocator, physical page pool and global mapping currently share one
  // process-wide device context. Reject multiple devices before starting RPCs.
  CHECK_EQ(devices.size(), 1) << "Distributed model memory requires exactly "
                                 "one local device per process";
  CHECK_GT(dp_size, 0) << "dp_size must be positive";
  CHECK(cluster_.servers().empty())
      << "Distributed model memory already set up";

  const auto& distributed_config = DistributedConfig::get_instance();
  CHECK_GT(distributed_config.nnodes(), 0) << "nnodes must be positive";
  CHECK_GE(distributed_config.node_rank(), 0);
  CHECK_LT(distributed_config.node_rank(), distributed_config.nnodes());
  const int32_t world_size = distributed_config.nnodes();
  CHECK_EQ(world_size % dp_size, 0)
      << "world_size must be divisible by dp_size";

  const int32_t tp_size = world_size / dp_size;
  cluster_.configure(world_size, dp_size, tp_size);

  std::shared_ptr<CollectiveService> collective_service;
  if (distributed_config.node_rank() == 0) {
    collective_service = std::make_shared<CollectiveService>(world_size);
    XllmServer* collective_server =
        ServerRegistry::get_instance().register_server(
            cluster_.collective_server_name());
    CHECK(collective_server->start(collective_service,
                                   master_node_addr,
                                   cluster_.collective_server_name()))
        << "Failed to start model memory collective server on address: "
        << master_node_addr;
  }

  cluster_.servers().emplace_back(std::make_unique<ModelMemoryDistServer>(
      /*local_rank=*/0, master_node_addr, devices.front(), options));
  CHECK(cluster_.servers().front()->wait_until_ready())
      << "Failed to start or register local model memory RPC server";

  if (collective_service == nullptr) {
    return;
  }
  auto model_memory_dist_addrs_map = collective_service->wait();

  auto& dp_group_clients = cluster_.dp_group_clients();
  auto& clients = cluster_.clients();
  clients.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    auto address = model_memory_dist_addrs_map.find(rank);
    CHECK(address != model_memory_dist_addrs_map.end())
        << "model memory RPC server did not connect to master node: rank "
        << rank;
    auto client = std::make_shared<ModelMemoryDistClient>(
        rank, address->second, devices.front());
    clients.emplace_back(client);
    const int32_t dp_rank = rank / tp_size;
    dp_group_clients[dp_rank].emplace_back(std::move(client));
  }

  LOG(INFO) << "Model memory RPC setup: world_size=" << world_size
            << ", dp_size=" << dp_size << ", tp_size=" << tp_size;
}

int64_t ModelMemoryManager::init_physical_page_pools(
    double max_memory_utilization,
    int64_t max_cache_size) {
  if (cluster_.world_size() <= 1) {
    // Single process single GPU, initialize locally
    Device device(dev_);
    device.set_device();

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    int64_t cache_size = available_memory;
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      cache_size -= buffer_memory;
    }
    if (max_cache_size > 0) {
      cache_size = std::min(cache_size, max_cache_size);
    }

    int64_t num_pages =
        cache_size /
        ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
    LOG(INFO) << "init_physical_page_pools (local): available_memory="
              << available_memory << ", total_memory=" << total_memory
              << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

    const size_t page_size = static_cast<size_t>(
        KVCacheConfig::get_instance().phy_page_granularity_size());
    PhysicalPagePool::get_instance().init(dev_, num_pages, page_size);

    // Initialize GlobalMemoryRegion after PhysicalPagePool
    GlobalMemoryRegion::get_instance().init(dev_);
    LOG(INFO) << "GlobalMemoryRegion initialized (local)";

    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(cluster_.clients().size());
  for (auto& client : cluster_.clients()) {
    memory_futures.push_back(client->get_memory_info_async());
  }

  // Wait for all memory info responses
  auto memory_results = folly::collectAll(memory_futures).get();

  int64_t min_available_memory = std::numeric_limits<int64_t>::max();
  int64_t min_total_memory = std::numeric_limits<int64_t>::max();

  for (size_t i = 0; i < memory_results.size(); ++i) {
    if (!memory_results[i].hasValue()) {
      LOG(ERROR) << "Failed to get memory info from worker: " << i;
      return 0;
    }
    auto& info = memory_results[i].value();
    if (info.available_memory == 0 && info.total_memory == 0) {
      LOG(ERROR) << "Worker " << i << " returned invalid memory info";
      return 0;
    }

    LOG(INFO) << "Worker #" << i
              << ": available_memory=" << info.available_memory
              << ", total_memory=" << info.total_memory;

    min_available_memory =
        std::min(min_available_memory, info.available_memory);
    min_total_memory = std::min(min_total_memory, info.total_memory);
  }

  // Step 2: Calculate num_pages based on min available memory
  int64_t cache_size = min_available_memory;
  if (max_memory_utilization < 1.0) {
    const int64_t buffer_memory =
        min_total_memory * (1.0 - max_memory_utilization);
    cache_size -= buffer_memory;
  }
  if (max_cache_size > 0) {
    cache_size = std::min(cache_size, max_cache_size);
  }

  int64_t num_pages =
      cache_size /
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  LOG(INFO) << "init_physical_page_pools: min_available_memory="
            << min_available_memory << ", min_total_memory=" << min_total_memory
            << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

  if (num_pages <= 0) {
    LOG(ERROR) << "Insufficient memory for PhysicalPagePool";
    return 0;
  }

  // Step 3: Broadcast InitPhysicalPagePool to all workers
  std::vector<folly::SemiFuture<bool>> init_futures;
  init_futures.reserve(cluster_.clients().size());
  for (auto& client : cluster_.clients()) {
    init_futures.push_back(client->init_physical_page_pool_async(num_pages));
  }

  // Wait for all init responses
  auto init_results = folly::collectAll(init_futures).get();
  for (size_t i = 0; i < init_results.size(); ++i) {
    if (!init_results[i].hasValue() || !init_results[i].value()) {
      LOG(ERROR) << "Failed to init PhysicalPagePool on worker: " << i;
      return 0;
    }
  }

  LOG(INFO) << "Successfully initialized PhysicalPagePool on all "
            << cluster_.world_size() << " workers with " << num_pages
            << " pages each";
  return num_pages;
}

// ============== Model Parallel Strategy ==============

void ModelMemoryManager::set_model_parallel_strategy(
    const std::string& model_id,
    int32_t dp_size,
    int32_t tp_size) {
  std::lock_guard<std::mutex> lock(mtx_);
  model_parallel_strategies_[model_id] = {dp_size, tp_size};
  LOG(INFO) << "Set model parallel strategy for " << model_id
            << ": dp_size=" << dp_size << ", tp_size=" << tp_size;
}

std::pair<int32_t, int32_t> ModelMemoryManager::get_model_parallel_strategy(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  const auto it = model_parallel_strategies_.find(model_id);
  if (it != model_parallel_strategies_.end() && it->second.first > 0 &&
      it->second.second > 0) {
    return it->second;
  }
  // Fallback to global values
  return {cluster_.dp_size(), cluster_.tp_size()};
}

// ============== Broadcast Operations ==============

bool ModelMemoryManager::broadcast_map_to_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (cluster_.world_size() <= 1) {
    // Single process single GPU, just map locally
    return map_to_kv_tensors(model_id, offsets);
  }

  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);

  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, model_dp_size) << "dp_rank must be < model_dp_size";

  // Calculate worker range for this DP group based on model's parallel strategy
  // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1, ...]
  int32_t start_rank = dp_rank * model_tp_size;
  int32_t end_rank = start_rank + model_tp_size;

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_tp_size);
  for (int32_t r = start_rank;
       r < end_rank && r < static_cast<int32_t>(cluster_.clients().size());
       ++r) {
    futures.push_back(
        cluster_.clients()[r]->map_to_kv_tensors_async(model_id, offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool ModelMemoryManager::broadcast_unmap_from_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (cluster_.world_size() <= 1) {
    // Single process single GPU, just unmap locally
    return unmap_from_kv_tensors(model_id, offsets);
  }

  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);

  CHECK_GE(dp_rank, 0) << "dp_rank must be >= 0";
  CHECK_LT(dp_rank, model_dp_size) << "dp_rank must be < model_dp_size";

  // Calculate worker range for this DP group based on model's parallel strategy
  // Workers are organized as: [dp0_tp0, dp0_tp1, ..., dp1_tp0, dp1_tp1, ...]
  int32_t start_rank = dp_rank * model_tp_size;
  int32_t end_rank = start_rank + model_tp_size;

  // Broadcast to workers in this DP group via RPC asynchronously
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_tp_size);
  for (int32_t r = start_rank;
       r < end_rank && r < static_cast<int32_t>(cluster_.clients().size());
       ++r) {
    futures.push_back(
        cluster_.clients()[r]->unmap_from_kv_tensors_async(model_id, offsets));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      return false;
    }
  }
  return true;
}

bool ModelMemoryManager::broadcast_alloc_weight_pages(
    const std::string& model_id,
    size_t num_pages) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (model_world_size <= 1) {
    WeightPageReservation reservation =
        page_coordinator_.reserve_weight_pages(num_pages);
    if (reservation.page_ids.empty()) {
      return false;
    }

    if (reservation.contiguous_start >= 0) {
      return record_weight_allocation(
          model_id, reservation.contiguous_start, num_pages);
    }
    return record_weight_fallback_allocation(model_id, reservation.page_ids);
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(cluster_.clients().size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.push_back(
        cluster_.clients()[i]->alloc_weight_pages_async(model_id, num_pages));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      LOG(ERROR) << "broadcast_alloc_weight_pages failed for model "
                 << model_id;
      return false;
    }
  }

  LOG(INFO) << "broadcast_alloc_weight_pages success: model=" << model_id
            << ", num_pages=" << num_pages << ", num_workers=" << num_workers;
  return true;
}

bool ModelMemoryManager::broadcast_free_weight_pages(
    const std::string& model_id) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (model_world_size <= 1) {
    // Single process: free locally
    free_weight(model_id);
    return true;
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(cluster_.clients().size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.push_back(cluster_.clients()[i]->free_weight_pages_async(model_id));
  }

  // Wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.hasValue() || !result.value()) {
      LOG(ERROR) << "broadcast_free_weight_pages failed for model " << model_id;
      return false;
    }
  }

  LOG(INFO) << "broadcast_free_weight_pages success: model=" << model_id
            << ", num_workers=" << num_workers;
  return true;
}

// ============== KV Cache Interfaces ==============

std::vector<torch::Tensor> ModelMemoryManager::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mtx_);
  return kv_cache_regions_[model_id].create_k_tensors(
      dims,
      dtype,
      num_layers,
      dev_,
      KVCacheConfig::get_instance().phy_page_granularity_size());
}

std::vector<torch::Tensor> ModelMemoryManager::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mtx_);
  return kv_cache_regions_[model_id].create_v_tensors(
      dims,
      dtype,
      num_layers,
      dev_,
      KVCacheConfig::get_instance().phy_page_granularity_size());
}

bool ModelMemoryManager::map_to_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = kv_cache_regions_.find(model_id);
  if (it == kv_cache_regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return false;
  }
  return it->second.map(offsets);
}

bool ModelMemoryManager::unmap_from_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = kv_cache_regions_.find(model_id);
  if (it == kv_cache_regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return false;
  }
  return it->second.unmap(offsets);
}

bool ModelMemoryManager::record_weight_allocation(const std::string& model_id,
                                                  page_id_t start_page_id,
                                                  size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto& global_memory_region = GlobalMemoryRegion::get_instance();
  void* base_ptr = global_memory_region.get_vaddr_by_page_id(start_page_id);
  auto& pool = PhysicalPagePool::get_instance();
  CHECK_GE(start_page_id, 0);
  CHECK_LT(static_cast<size_t>(start_page_id), pool.num_total());
  CHECK_LE(num_pages, pool.num_total() - static_cast<size_t>(start_page_id));
  if (base_ptr == nullptr || num_pages == 0) {
    LOG(ERROR)
        << "ModelMemoryManager: invalid GlobalMemoryRegion region for model "
        << model_id;
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.emplace_back(start_page_id + static_cast<page_id_t>(i));
    }
    pool.release_reserved_pages(page_ids);
    return false;
  }

  auto& weight = weight_store_.get_or_create(model_id);
  size_t page_size = global_memory_region.page_size();
  std::vector<WeightSegment> segments = {
      {static_cast<uint64_t>(start_page_id) * page_size,
       static_cast<uint64_t>(num_pages) * page_size}};
  weight.set_contiguous(
      start_page_id, num_pages, base_ptr, std::move(segments));

  LOG(INFO) << "ModelMemoryManager: recorded weight allocation for model "
            << model_id << ", start_page=" << start_page_id
            << ", num_pages=" << num_pages << ", base_ptr=" << base_ptr;
  return true;
}

bool ModelMemoryManager::record_weight_fallback_allocation(
    const std::string& model_id,
    const std::vector<page_id_t>& page_ids) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Create MappedMemoryRegion with the non-contiguous preallocated pages
  auto weight_region = std::make_unique<MappedMemoryRegion>(
      page_ids,
      torch::kBFloat16,
      dev_,
      static_cast<size_t>(
          KVCacheConfig::get_instance().phy_page_granularity_size()));
  auto& global_memory_region = GlobalMemoryRegion::get_instance();
  if (is_null_vir_ptr(weight_region->vaddr()) ||
      !global_memory_region.is_initialized()) {
    LOG(ERROR)
        << "ModelMemoryManager: failed to create MappedMemoryRegion for model "
        << model_id;
    // The MappedMemoryRegion destructor returns its reserved pages to the pool.
    return false;
  }

  auto& weight = weight_store_.get_or_create(model_id);
  // Preserve logical mapping order; merge only forward-adjacent pages.
  size_t page_size = global_memory_region.page_size();
  const auto raw_segments = make_weight_transfer_segments(page_ids, page_size);
  std::vector<WeightSegment> segments;
  segments.reserve(raw_segments.size());
  for (const auto& [offset, size] : raw_segments) {
    segments.emplace_back(offset, size);
  }
  weight.set_fragmented(
      std::move(weight_region), page_ids.size(), std::move(segments));

  LOG(INFO)
      << "ModelMemoryManager: recorded mapped memory allocation for model "
      << model_id << ", num_pages=" << page_ids.size()
      << ", base_ptr=" << weight.base_ptr()
      << ", weight_segments=" << weight.segments().size() << " (fallback mode)";
  return true;
}

bool ModelMemoryManager::allocate_weight(const std::string& model_id,
                                         void*& ptr,
                                         size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* weight = weight_store_.find(model_id);
  if (weight == nullptr || weight->base_ptr() == nullptr) {
    LOG(ERROR) << "No pre-allocated weight region for model " << model_id;
    return false;
  }

  auto& global_memory_region = GlobalMemoryRegion::get_instance();
  if (!weight->allocate(ptr, size, global_memory_region.page_size())) {
    LOG(ERROR) << "Not enough space in weight region for model " << model_id
               << ": requested " << size;
    return false;
  }

  VLOG(1) << "ModelMemoryManager: allocated " << size << " bytes for model "
          << model_id << ", ptr=" << ptr;

  return true;
}

// ============== Internal Helpers ==============

void ModelMemoryManager::init_device_() {
  Device device(dev_);
  device.set_device();
  device.init_device_context();

  const size_t chunk_sz = vmm::get_recommended_granularity(dev_.index());
  KVCacheConfig::get_instance().phy_page_granularity_size(chunk_sz);
  LOG(INFO) << "Device initialized with granularity size: " << chunk_sz
            << " bytes";
}

size_t ModelMemoryManager::free_weight(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* weight = weight_store_.find(model_id);
  if (weight == nullptr || weight->num_pages() == 0) {
    LOG(WARNING) << "No weight allocation found for model " << model_id;
    return 0;
  }

  size_t num_pages = weight->num_pages();

  // Handle MappedMemoryRegion fallback case
  if (weight->is_fragmented()) {
    // MappedMemoryRegion's destructor will unmap and free pages
    weight->reset();
    LOG(INFO) << "Freed " << num_pages
              << " weight pages (MappedMemoryRegion fallback) for model "
              << model_id;
  } else {
    // Normal path: free contiguous pages from GlobalMemoryRegion
    page_id_t start_page = weight->start_page_id();

    // Build page_ids vector and free via PhysicalPagePool
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.push_back(start_page + static_cast<page_id_t>(i));
    }

    WeightPageReservation reservation;
    reservation.contiguous_start = start_page;
    reservation.page_ids = std::move(page_ids);
    page_coordinator_.release_weight_pages(reservation);

    LOG(INFO) << "Freed " << num_pages << " weight pages for model "
              << model_id;
  }

  // Clear weight allocation record
  weight->reset();

  return num_pages;
}

// ============== PD Disaggregation Support (MappedMemoryRegion Mode)
// ==============

std::pair<uint64_t, uint64_t> ModelMemoryManager::get_global_offsets_for_block(
    const std::string& model_id,
    int64_t layer_id,
    int64_t block_id,
    size_t block_size) {
  constexpr uint64_t kInvalidOffset = UINT64_MAX;
  std::lock_guard<std::mutex> lock(mtx_);

  const auto it = kv_cache_regions_.find(model_id);
  if (it == kv_cache_regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return {kInvalidOffset, kInvalidOffset};
  }
  if (!GlobalMemoryRegion::get_instance().is_initialized()) {
    LOG(ERROR) << "GlobalMemoryRegion not initialized";
    return {kInvalidOffset, kInvalidOffset};
  }
  return it->second.get_global_offsets_for_block(
      layer_id, block_id, block_size);
}

bool ModelMemoryManager::get_kv_cache_offsets(
    int32_t dp_rank,
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    uint64_t block_size_bytes,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) {
  // Cache mappings in the master's DP group have identical pool-relative
  // offsets and can be queried locally.
  if (dp_rank == 0) {
    const std::optional<int64_t> model_num_layers =
        get_kv_cache_num_layers(model_id);
    if (!model_num_layers.has_value()) {
      LOG(ERROR) << "Model " << model_id << " not found for local calculation";
      return false;
    }

    const int64_t num_layers = model_num_layers.value();
    layer_offsets.resize(num_layers);

    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      std::vector<uint64_t> k_offsets;
      std::vector<uint64_t> v_offsets;
      k_offsets.reserve(block_ids.size());
      v_offsets.reserve(block_ids.size());

      for (int32_t block_id : block_ids) {
        auto [k_offset, v_offset] = get_global_offsets_for_block(
            model_id, layer_id, block_id, block_size_bytes);
        if (k_offset == UINT64_MAX || v_offset == UINT64_MAX) {
          LOG(ERROR) << "Failed to get local offsets for block " << block_id
                     << " at layer " << layer_id;
          return false;
        }
        k_offsets.push_back(k_offset);
        v_offsets.push_back(v_offset);
      }
      layer_offsets[layer_id] = {std::move(k_offsets), std::move(v_offsets)};
    }

    VLOG(1) << "get_kv_cache_offsets (local): model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
    return true;
  }

  if (dp_rank < 0 ||
      dp_rank >= static_cast<int32_t>(cluster_.dp_group_clients().size())) {
    LOG(ERROR) << "Invalid dp_rank: " << dp_rank << ", dp_group_clients.size()="
               << cluster_.dp_group_clients().size();
    return false;
  }

  const auto& clients = cluster_.dp_group_clients()[dp_rank];
  if (clients.empty()) {
    LOG(ERROR) << "No clients in dp_group " << dp_rank;
    return false;
  }

  // Call the first worker in the DP group (all workers in the same DP group
  // should have the same physical page mapping)
  auto& client = clients[0];
  auto future =
      client->get_kv_cache_offsets_async(model_id, block_ids, block_size_bytes);

  layer_offsets = std::move(future).get();
  if (layer_offsets.empty()) {
    LOG(ERROR) << "get_kv_cache_offsets failed for dp_rank=" << dp_rank
               << ", model_id=" << model_id;
    return false;
  }

  VLOG(1) << "get_kv_cache_offsets: dp_rank=" << dp_rank
          << ", model_id=" << model_id << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();

  return true;
}

// ============== ETCD Information Support ==============

std::vector<WeightSegment> ModelMemoryManager::get_model_weight_segments(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  const auto* weight = weight_store_.find(model_id);
  if (weight == nullptr) {
    return {};
  }
  return weight->segments();
}

std::unordered_map<std::string, std::vector<WeightSegment>>
ModelMemoryManager::get_all_model_weight_segments() const {
  std::lock_guard<std::mutex> lock(mtx_);
  std::unordered_map<std::string, std::vector<WeightSegment>> result;

  for (const auto& [model_id, weight] : weight_store_.models()) {
    if (!weight.segments().empty()) {
      result[model_id] = weight.segments();
    }
  }

  return result;
}

}  // namespace xllm

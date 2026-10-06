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

#include "core/framework/xtensor/xtensor_allocator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "core/distributed_runtime/collective_service.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/phy_page.h"
#include "core/framework/xtensor/phy_page_pool.h"
#include "core/framework/xtensor/weight_transfer_segments.h"
#include "core/framework/xtensor/xtensor.h"
#include "core/platform/device.h"
#include "core/platform/vmm_api.h"
#include "server/xllm_server_registry.h"

namespace xllm {

XTensorAllocator::~XTensorAllocator() {
  if (!initialized_) {
    return;
  }

  // Stop collective server if running
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }

  destroy();
}

void XTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  model_tensors_.clear();
  xtensor_dist_clients_.clear();
  xtensor_dist_servers_.clear();
  initialized_ = false;
}

void XTensorAllocator::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (initialized_) {
    LOG(WARNING) << "XTensorAllocator already initialized, ignoring re-init";
    return;
  }

  dev_ = device;
  init_device_();
  initialized_ = true;
}

ModelTensors& XTensorAllocator::get_or_create_model_tensors(
    const std::string& model_id) {
  // Note: caller must hold mtx_
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    model_tensors_[model_id] = ModelTensors{};
    VLOG(1) << "Auto-created model tensors entry for: " << model_id;
  }
  return model_tensors_[model_id];
}

ModelTensors* XTensorAllocator::get_model_tensors(const std::string& model_id) {
  // Note: caller must hold mtx_
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

// ============== Multi-node Setup ==============

void XTensorAllocator::setup_multi_node_xtensor_dist(
    const xtensor::Options& options,
    const std::string& master_node_addr,
    int32_t dp_size) {
  const auto& devices = options.devices();
  // The allocator, physical page pool and global mapping currently share one
  // process-wide device context. Reject multiple devices before starting RPCs.
  CHECK_EQ(devices.size(), 1)
      << "Distributed XTensor requires exactly one local device per process";
  CHECK_GT(dp_size, 0) << "dp_size must be positive";
  CHECK(xtensor_dist_servers_.empty()) << "Distributed XTensor already set up";

  const auto& distributed_config = DistributedConfig::get_instance();
  CHECK_GT(distributed_config.nnodes(), 0) << "nnodes must be positive";
  CHECK_GE(distributed_config.node_rank(), 0);
  CHECK_LT(distributed_config.node_rank(), distributed_config.nnodes());
  const int32_t world_size = distributed_config.nnodes();
  CHECK_EQ(world_size % dp_size, 0)
      << "world_size must be divisible by dp_size";

  world_size_ = world_size;
  dp_size_ = dp_size;
  tp_size_ = world_size_ / dp_size_;
  collective_server_name_ = "XTensorDistCollectiveServer";

  std::shared_ptr<CollectiveService> collective_service;
  if (distributed_config.node_rank() == 0) {
    collective_service = std::make_shared<CollectiveService>(world_size_);
    XllmServer* collective_server =
        ServerRegistry::get_instance().register_server(collective_server_name_);
    CHECK(collective_server->start(
        collective_service, master_node_addr, collective_server_name_))
        << "Failed to start XTensor collective server on address: "
        << master_node_addr;
  }

  xtensor_dist_servers_.emplace_back(std::make_unique<XTensorDistServer>(
      /*local_rank=*/0, master_node_addr, devices.front(), options));
  CHECK(xtensor_dist_servers_.front()->wait_until_ready())
      << "Failed to start or register local XTensor dist server";

  if (collective_service == nullptr) {
    return;
  }
  auto xtensor_dist_addrs_map = collective_service->wait();

  dp_group_clients_.resize(dp_size_);
  for (auto& clients : dp_group_clients_) {
    clients.reserve(tp_size_);
  }
  xtensor_dist_clients_.reserve(world_size_);
  for (int32_t rank = 0; rank < world_size_; ++rank) {
    auto address = xtensor_dist_addrs_map.find(rank);
    CHECK(address != xtensor_dist_addrs_map.end())
        << "XTensor dist server did not connect to master node: rank " << rank;
    auto client = std::make_shared<XTensorDistClient>(
        rank, address->second, devices.front());
    xtensor_dist_clients_.emplace_back(client);
    const int32_t dp_rank = rank / tp_size_;
    dp_group_clients_[dp_rank].emplace_back(std::move(client));
  }

  LOG(INFO) << "XTensor dist setup: world_size=" << world_size_
            << ", dp_size=" << dp_size_ << ", tp_size=" << tp_size_;
}

int64_t XTensorAllocator::init_phy_page_pools(double max_memory_utilization,
                                              int64_t max_cache_size) {
  if (world_size_ <= 1) {
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
    LOG(INFO) << "init_phy_page_pools (local): available_memory="
              << available_memory << ", total_memory=" << total_memory
              << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

    PhyPagePool::get_instance().init(dev_, num_pages);

    // Initialize GlobalXTensor after PhyPagePool
    GlobalXTensor::get_instance().init(dev_);
    LOG(INFO) << "GlobalXTensor initialized (local)";

    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
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
  LOG(INFO) << "init_phy_page_pools: min_available_memory="
            << min_available_memory << ", min_total_memory=" << min_total_memory
            << ", cache_size=" << cache_size << ", num_pages=" << num_pages;

  if (num_pages <= 0) {
    LOG(ERROR) << "Insufficient memory for PhyPagePool";
    return 0;
  }

  // Step 3: Broadcast InitPhyPagePool to all workers
  std::vector<folly::SemiFuture<bool>> init_futures;
  init_futures.reserve(xtensor_dist_clients_.size());
  for (auto& client : xtensor_dist_clients_) {
    init_futures.push_back(client->init_phy_page_pool_async(num_pages));
  }

  // Wait for all init responses
  auto init_results = folly::collectAll(init_futures).get();
  for (size_t i = 0; i < init_results.size(); ++i) {
    if (!init_results[i].hasValue() || !init_results[i].value()) {
      LOG(ERROR) << "Failed to init PhyPagePool on worker: " << i;
      return 0;
    }
  }

  LOG(INFO) << "Successfully initialized PhyPagePool on all " << world_size_
            << " workers with " << num_pages << " pages each";
  return num_pages;
}

// ============== Model Parallel Strategy ==============

void XTensorAllocator::set_model_parallel_strategy(const std::string& model_id,
                                                   int32_t dp_size,
                                                   int32_t tp_size) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.dp_size = dp_size;
  tensors.tp_size = tp_size;
  LOG(INFO) << "Set model parallel strategy for " << model_id
            << ": dp_size=" << dp_size << ", tp_size=" << tp_size;
}

std::pair<int32_t, int32_t> XTensorAllocator::get_model_parallel_strategy(
    const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto* tensors = get_model_tensors(model_id);
  if (tensors && tensors->dp_size > 0 && tensors->tp_size > 0) {
    return {tensors->dp_size, tensors->tp_size};
  }
  // Fallback to global values
  return {dp_size_, tp_size_};
}

// ============== Broadcast Operations ==============

bool XTensorAllocator::broadcast_map_to_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
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
       r < end_rank && r < static_cast<int32_t>(xtensor_dist_clients_.size());
       ++r) {
    futures.push_back(
        xtensor_dist_clients_[r]->map_to_kv_tensors_async(model_id, offsets));
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

bool XTensorAllocator::broadcast_unmap_from_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<offset_t>& offsets) {
  if (world_size_ <= 1) {
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
       r < end_rank && r < static_cast<int32_t>(xtensor_dist_clients_.size());
       ++r) {
    futures.push_back(xtensor_dist_clients_[r]->unmap_from_kv_tensors_async(
        model_id, offsets));
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

bool XTensorAllocator::broadcast_alloc_weight_pages(const std::string& model_id,
                                                    size_t num_pages) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (model_world_size <= 1) {
    // Single process: allocate locally from PhyPagePool and record
    auto& pool = PhyPagePool::get_instance();

    // Try contiguous allocation first (from GlobalXTensor)
    page_id_t start_page = pool.allocate_contiguous_from_right(num_pages);
    if (start_page >= 0) {
      return record_weight_allocation(model_id, start_page, num_pages);
    }

    // Fallback: try non-contiguous allocation using XTensor
    LOG(WARNING) << "Contiguous allocation failed for " << num_pages
                 << " pages, trying non-contiguous fallback (XTensor)";

    std::vector<page_id_t> page_ids = pool.allocate_pages_from_right(num_pages);
    if (page_ids.empty()) {
      LOG(ERROR) << "Failed to allocate " << num_pages
                 << " weight pages (both contiguous and non-contiguous)";
      return false;
    }

    return record_weight_fallback_allocation(model_id, page_ids);
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers = std::min(
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.push_back(xtensor_dist_clients_[i]->alloc_weight_pages_async(
        model_id, num_pages));
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

bool XTensorAllocator::broadcast_free_weight_pages(
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
      model_world_size, static_cast<int32_t>(xtensor_dist_clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.push_back(
        xtensor_dist_clients_[i]->free_weight_pages_async(model_id));
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

std::vector<torch::Tensor> XTensorAllocator::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(model_id, dims, dtype, num_layers, "K");
}

std::vector<torch::Tensor> XTensorAllocator::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  return create_kv_tensors_impl_(model_id, dims, dtype, num_layers, "V");
}

std::vector<torch::Tensor> XTensorAllocator::create_kv_tensors_impl_(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    const char* name) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Get or create model tensors entry
  auto& model = get_or_create_model_tensors(model_id);

  // Select target tensors based on name
  std::vector<std::unique_ptr<XTensor>>* target_tensors = nullptr;
  if (strcmp(name, "K") == 0) {
    target_tensors = &model.k_tensors;
  } else if (strcmp(name, "V") == 0) {
    target_tensors = &model.v_tensors;
  } else {
    LOG(FATAL) << "Unknown tensor name: " << name;
  }

  CHECK(model.num_layers == 0 || model.num_layers == num_layers)
      << "Number of layers mismatch for model " << model_id;
  CHECK(target_tensors->empty())
      << name << " tensors already created for model " << model_id;
  CHECK(!dims.empty()) << name << " tensor dims cannot be empty";
  CHECK_GT(num_layers, 0);

  // Calculate size from dims and dtype
  size_t size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  for (int64_t dim : dims) {
    CHECK_GT(dim, 0);
    CHECK_LE(size,
             std::numeric_limits<size_t>::max() / static_cast<size_t>(dim));
    size *= static_cast<size_t>(dim);
  }

  size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  CHECK_GT(page_size, 0);
  CHECK_LE(size, std::numeric_limits<size_t>::max() - (page_size - 1));
  // Align size to page size (round up)
  if (size % page_size != 0) {
    size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
    LOG(WARNING) << name << " tensor size " << size
                 << " is not aligned to page size " << page_size
                 << ", aligning to " << aligned_size;
    size = aligned_size;
  }

  model.num_layers = num_layers;
  model.kv_tensor_size_per_layer = size;

  return create_tensors_internal_(
      size, dims, dtype, num_layers, *target_tensors);
}

bool XTensorAllocator::map_to_kv_tensors(const std::string& model_id,
                                         const std::vector<offset_t>& offsets) {
  std::unique_lock<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found";
    return false;
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "KV tensors not created for model " << model_id;
    return false;
  }

  CHECK_GT(tensors->num_layers, 0);
  CHECK_EQ(tensors->k_tensors.size(), tensors->num_layers);
  CHECK_EQ(tensors->v_tensors.size(), tensors->num_layers);
  CHECK_LE(offsets.size(),
           std::numeric_limits<size_t>::max() / tensors->k_tensors.size() / 2);
  std::vector<std::pair<XTensor*, offset_t>> targets;
  targets.reserve(offsets.size() * tensors->k_tensors.size() * 2);
  for (size_t i = 0; i < tensors->k_tensors.size(); ++i) {
    for (offset_t offset : offsets) {
      targets.emplace_back(tensors->k_tensors[i].get(), offset);
      targets.emplace_back(tensors->v_tensors[i].get(), offset);
    }
  }
  return XTensor::map_pages(targets);
}

bool XTensorAllocator::unmap_from_kv_tensors(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  std::unique_lock<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found";
    return false;
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "try to unmap from KV tensors when KV tensors are not created"
               << " for model " << model_id;
    return false;
  }

  CHECK_GT(tensors->num_layers, 0);
  CHECK_EQ(tensors->k_tensors.size(), tensors->num_layers);
  CHECK_EQ(tensors->v_tensors.size(), tensors->num_layers);
  for (size_t i = 0; i < tensors->k_tensors.size(); ++i) {
    const auto& k_tensor = tensors->k_tensors[i];
    const auto& v_tensor = tensors->v_tensors[i];
    for (offset_t offset : offsets) {
      if (offset < 0 || static_cast<size_t>(offset) >= k_tensor->size() ||
          static_cast<size_t>(offset) >= v_tensor->size() ||
          static_cast<size_t>(offset) % k_tensor->page_size() != 0 ||
          static_cast<size_t>(offset) % v_tensor->page_size() != 0) {
        LOG(ERROR) << "Invalid KV unmapping offset " << offset;
        return false;
      }
    }
  }

  for (size_t i = 0; i < tensors->k_tensors.size(); ++i) {
    for (offset_t offset : offsets) {
      CHECK(tensors->k_tensors[i]->unmap(offset));
      CHECK(tensors->v_tensors[i]->unmap(offset));
    }
  }
  return true;
}

bool XTensorAllocator::record_weight_allocation(const std::string& model_id,
                                                page_id_t start_page_id,
                                                size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto& global_xtensor = GlobalXTensor::get_instance();
  void* base_ptr = global_xtensor.get_vaddr_by_page_id(start_page_id);
  auto& pool = PhyPagePool::get_instance();
  CHECK_GE(start_page_id, 0);
  CHECK_LT(static_cast<size_t>(start_page_id), pool.num_total());
  CHECK_LE(num_pages, pool.num_total() - static_cast<size_t>(start_page_id));
  if (base_ptr == nullptr || num_pages == 0) {
    LOG(ERROR) << "XTensorAllocator: invalid GlobalXTensor region for model "
               << model_id;
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.emplace_back(start_page_id + static_cast<page_id_t>(i));
    }
    pool.free_weight_pages(page_ids);
    return false;
  }

  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.weight_start_page_id = start_page_id;
  tensors.weight_num_pages = num_pages;
  tensors.weight_base_ptr = base_ptr;
  tensors.weight_current_offset = 0;
  tensors.using_weight_xtensor = false;
  tensors.weight_xtensor.reset();

  // Populate weight_segments for P2P transfer support
  size_t page_size = global_xtensor.page_size();
  tensors.weight_segments.clear();
  tensors.weight_segments.push_back(
      {static_cast<uint64_t>(start_page_id) * page_size,
       static_cast<uint64_t>(num_pages) * page_size});

  LOG(INFO) << "XTensorAllocator: recorded weight allocation for model "
            << model_id << ", start_page=" << start_page_id
            << ", num_pages=" << num_pages << ", base_ptr=" << base_ptr;
  return true;
}

bool XTensorAllocator::record_weight_fallback_allocation(
    const std::string& model_id,
    const std::vector<page_id_t>& page_ids) {
  std::lock_guard<std::mutex> lock(mtx_);

  // Create XTensor with the non-contiguous preallocated pages
  auto weight_xtensor =
      std::make_unique<XTensor>(page_ids, torch::kBFloat16, dev_);
  auto& global_xtensor = GlobalXTensor::get_instance();
  if (is_null_vir_ptr(weight_xtensor->vaddr()) ||
      !global_xtensor.is_initialized()) {
    LOG(ERROR) << "XTensorAllocator: failed to create XTensor for model "
               << model_id;
    // The XTensor destructor returns its reserved pages to the pool.
    return false;
  }

  auto& tensors = get_or_create_model_tensors(model_id);
  tensors.weight_xtensor = std::move(weight_xtensor);
  tensors.using_weight_xtensor = true;
  tensors.weight_num_pages = page_ids.size();
  tensors.weight_base_ptr =
      vir_ptr_to_void_ptr(tensors.weight_xtensor->vaddr());
  tensors.weight_current_offset = 0;
  tensors.weight_start_page_id = -1;  // Not applicable for non-contiguous

  // Populate weight_segments for P2P transfer support
  // Preserve logical mapping order; merge only forward-adjacent pages.
  size_t page_size = global_xtensor.page_size();
  tensors.weight_segments.clear();
  auto segments = make_weight_transfer_segments(page_ids, page_size);
  tensors.weight_segments.reserve(segments.size());
  for (const auto& [offset, size] : segments) {
    tensors.weight_segments.emplace_back(offset, size);
  }

  LOG(INFO) << "XTensorAllocator: recorded XTensor allocation for model "
            << model_id << ", num_pages=" << page_ids.size()
            << ", base_ptr=" << tensors.weight_base_ptr
            << ", weight_segments=" << tensors.weight_segments.size()
            << " (fallback mode)";
  return true;
}

bool XTensorAllocator::allocate_weight(const std::string& model_id,
                                       void*& ptr,
                                       size_t size) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors || tensors->weight_base_ptr == nullptr) {
    LOG(ERROR) << "No pre-allocated weight region for model " << model_id;
    return false;
  }

  // Use XTensor's allocate if in fallback mode
  if (tensors->using_weight_xtensor && tensors->weight_xtensor) {
    if (!tensors->weight_xtensor->allocate(ptr, size)) {
      LOG(ERROR) << "XTensor::allocate failed for model " << model_id;
      return false;
    }
    tensors->weight_current_offset = tensors->weight_xtensor->alloc_offset();
    VLOG(1) << "XTensorAllocator: allocated " << size
            << " bytes via XTensor for model " << model_id << ", ptr=" << ptr;
    return true;
  }

  // Normal path: allocate from GlobalXTensor
  auto& global_xtensor = GlobalXTensor::get_instance();
  size_t region_size = tensors->weight_num_pages * global_xtensor.page_size();

  // Check if there's enough space in pre-allocated region
  CHECK_LE(tensors->weight_current_offset, region_size);
  if (size > region_size - tensors->weight_current_offset) {
    LOG(ERROR) << "Not enough space in weight region for model " << model_id
               << ": requested " << size << ", available "
               << (region_size - tensors->weight_current_offset);
    return false;
  }

  // Allocate from base + current offset
  ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(tensors->weight_base_ptr) +
      tensors->weight_current_offset);

  tensors->weight_current_offset += size;

  VLOG(1) << "XTensorAllocator: allocated " << size << " bytes for model "
          << model_id << ", ptr=" << ptr;

  return true;
}

// ============== Internal Helpers ==============

std::vector<torch::Tensor> XTensorAllocator::create_tensors_internal_(
    size_t size,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers,
    std::vector<std::unique_ptr<XTensor>>& tensors_out) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(num_layers);
  tensors_out.reserve(num_layers);

  for (int64_t i = 0; i < num_layers; i++) {
    auto xtensor = std::make_unique<XTensor>(size, dtype, dev_);
    tensors.push_back(xtensor->to_torch_tensor(0, dims));
    tensors_out.push_back(std::move(xtensor));
  }
  return tensors;
}

void XTensorAllocator::init_device_() {
  Device device(dev_);
  device.set_device();
  device.init_device_context();

  // Create a dummy PhyPage to initialize the granularity size
  // This will set
  // ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size()
  auto dummy_page = std::make_shared<PhyPage>(dev_);

  size_t chunk_sz =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  LOG(INFO) << "Device initialized with granularity size: " << chunk_sz
            << " bytes";
}

size_t XTensorAllocator::free_weight(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors || tensors->weight_num_pages == 0) {
    LOG(WARNING) << "No weight allocation found for model " << model_id;
    return 0;
  }

  size_t num_pages = tensors->weight_num_pages;

  // Handle XTensor fallback case
  if (tensors->using_weight_xtensor && tensors->weight_xtensor) {
    // XTensor's destructor will unmap and free pages
    tensors->weight_xtensor.reset();
    tensors->using_weight_xtensor = false;
    LOG(INFO) << "Freed " << num_pages
              << " weight pages (XTensor fallback) for model " << model_id;
  } else {
    // Normal path: free contiguous pages from GlobalXTensor
    page_id_t start_page = tensors->weight_start_page_id;

    // Build page_ids vector and free via PhyPagePool
    std::vector<page_id_t> page_ids;
    page_ids.reserve(num_pages);
    for (size_t i = 0; i < num_pages; ++i) {
      page_ids.push_back(start_page + static_cast<page_id_t>(i));
    }

    auto& pool = PhyPagePool::get_instance();
    pool.free_weight_pages(page_ids);

    LOG(INFO) << "Freed " << num_pages << " weight pages for model "
              << model_id;
  }

  // Clear weight allocation record
  tensors->weight_start_page_id = -1;
  tensors->weight_num_pages = 0;
  tensors->weight_base_ptr = nullptr;
  tensors->weight_current_offset = 0;
  tensors->weight_segments.clear();

  return num_pages;
}

// ============== PD Disaggregation Support (XTensor Mode) ==============

std::pair<uint64_t, uint64_t> XTensorAllocator::get_global_offsets_for_block(
    const std::string& model_id,
    int64_t layer_id,
    int64_t block_id,
    size_t block_size) {
  constexpr uint64_t INVALID_OFFSET = UINT64_MAX;

  std::lock_guard<std::mutex> lock(mtx_);

  auto* tensors = get_model_tensors(model_id);
  if (!tensors) {
    LOG(ERROR) << "Model " << model_id << " not found for offset calculation";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  if (layer_id < 0 || layer_id >= tensors->num_layers) {
    LOG(ERROR) << "Invalid layer_id " << layer_id << " for model " << model_id
               << " (num_layers=" << tensors->num_layers << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  if (tensors->k_tensors.empty() || tensors->v_tensors.empty()) {
    LOG(ERROR) << "KV tensors not created for model " << model_id;
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(ERROR) << "GlobalXTensor not initialized";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Calculate the offset within the XTensor for this block
  // The offset must be aligned to page_size
  size_t page_size =
      ::xllm::KVCacheConfig::get_instance().phy_page_granularity_size();
  offset_t local_offset =
      static_cast<offset_t>((block_id * block_size / page_size) * page_size);

  // Get K tensor's physical page_id at this offset
  auto* k_xtensor = tensors->k_tensors[layer_id].get();
  page_id_t k_page_id = k_xtensor->get_phy_page_id(local_offset);
  if (k_page_id < 0) {
    LOG(ERROR) << "K cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Get V tensor's physical page_id at this offset
  auto* v_xtensor = tensors->v_tensors[layer_id].get();
  page_id_t v_page_id = v_xtensor->get_phy_page_id(local_offset);
  if (v_page_id < 0) {
    LOG(ERROR) << "V cache block " << block_id << " at layer " << layer_id
               << " is not mapped (local_offset=" << local_offset << ")";
    return {INVALID_OFFSET, INVALID_OFFSET};
  }

  // Calculate GlobalXTensor offsets using page_id
  // GlobalXTensor offset = page_id * page_size + (block offset within page)
  size_t offset_within_page = (block_id * block_size) % page_size;

  uint64_t k_global_offset =
      static_cast<uint64_t>(k_page_id) * page_size + offset_within_page;
  uint64_t v_global_offset =
      static_cast<uint64_t>(v_page_id) * page_size + offset_within_page;

  VLOG(2) << "get_global_offsets_for_block: model=" << model_id
          << ", layer=" << layer_id << ", block=" << block_id
          << ", block_size=" << block_size << ", k_page_id=" << k_page_id
          << ", v_page_id=" << v_page_id << ", k_offset=" << k_global_offset
          << ", v_offset=" << v_global_offset;

  return {k_global_offset, v_global_offset};
}

bool XTensorAllocator::get_xtensor_offsets(
    int32_t dp_rank,
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    uint64_t block_size_bytes,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) {
  // The offsets of the xtensor in the same DP group as the master worker are
  // identical, so there is no need to fetch them via RPC.
  if (dp_rank == 0) {
    // Get model tensors to determine num_layers
    auto* tensors = get_model_tensors(model_id);
    if (!tensors) {
      LOG(ERROR) << "Model " << model_id << " not found for local calculation";
      return false;
    }

    int64_t num_layers = tensors->num_layers;
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

    VLOG(1) << "get_xtensor_offsets (local): model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
    return true;
  }

  if (dp_rank < 0 ||
      dp_rank >= static_cast<int32_t>(dp_group_clients_.size())) {
    LOG(ERROR) << "Invalid dp_rank: " << dp_rank
               << ", dp_group_clients_.size()=" << dp_group_clients_.size();
    return false;
  }

  const auto& clients = dp_group_clients_[dp_rank];
  if (clients.empty()) {
    LOG(ERROR) << "No clients in dp_group " << dp_rank;
    return false;
  }

  // Call the first worker in the DP group (all workers in the same DP group
  // should have the same physical page mapping)
  auto& client = clients[0];
  auto future =
      client->get_xtensor_offsets_async(model_id, block_ids, block_size_bytes);

  layer_offsets = std::move(future).get();
  if (layer_offsets.empty()) {
    LOG(ERROR) << "get_xtensor_offsets failed for dp_rank=" << dp_rank
               << ", model_id=" << model_id;
    return false;
  }

  VLOG(1) << "get_xtensor_offsets: dp_rank=" << dp_rank
          << ", model_id=" << model_id << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();

  return true;
}

// ============== ETCD Information Support ==============

std::vector<WeightSegment> XTensorAllocator::get_model_weight_segments(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = model_tensors_.find(model_id);
  if (it == model_tensors_.end()) {
    return {};
  }
  return it->second.weight_segments;
}

std::unordered_map<std::string, std::vector<WeightSegment>>
XTensorAllocator::get_all_model_weight_segments() const {
  std::lock_guard<std::mutex> lock(mtx_);
  std::unordered_map<std::string, std::vector<WeightSegment>> result;

  for (const auto& [model_id, tensors] : model_tensors_) {
    if (!tensors.weight_segments.empty()) {
      result[model_id] = tensors.weight_segments;
    }
  }

  return result;
}

}  // namespace xllm

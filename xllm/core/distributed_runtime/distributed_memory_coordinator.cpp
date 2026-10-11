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

#include "core/distributed_runtime/distributed_memory_coordinator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>

#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/worker_memory_rpc_client.h"
#include "core/distributed_runtime/worker_memory_rpc_server.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/storage/paged_kv_cache_page_allocator.h"
#include "core/platform/device.h"
#include "core/runtime/worker_memory_resources.h"
#include "server/xllm_server_registry.h"

namespace xllm {

DistributedMemoryCoordinator::DistributedMemoryCoordinator() {
  // These singletons must outlive the coordinator and its RPC shutdown.
  WorkerMemoryResources::get_instance();
  ServerRegistry::get_instance();
  kv_cache_pages_ = std::make_unique<PagedKVCachePageAllocator>(*this);
}

DistributedMemoryCoordinator::~DistributedMemoryCoordinator() { destroy(); }

void DistributedMemoryCoordinator::destroy() {
  // Drain background KV mappings while their backend and RPCs are still alive.
  kv_cache_pages_.reset();
  // Server destruction stops RPC handling and joins each serving thread before
  // worker-local resources or clients are released.
  servers_.clear();
  XllmServer* collective_server =
      ServerRegistry::get_instance().try_get_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();
    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  model_parallel_strategies_.clear();
  clients_.clear();
  world_size_ = 0;
  dp_size_ = 1;
  tp_size_ = 1;
  auto& resources = WorkerMemoryResources::get_instance();
  if (resources.is_initialized()) {
    resources.destroy();
  }
}

void DistributedMemoryCoordinator::configure_workers(int32_t world_size,
                                                     int32_t dp_size,
                                                     int32_t tp_size) {
  world_size_ = world_size;
  dp_size_ = dp_size;
  tp_size_ = tp_size;
  clients_.clear();
  clients_.reserve(world_size_);
}

void DistributedMemoryCoordinator::setup_worker_memory_rpc(
    const WorkerMemoryRpcOptions& options,
    const std::string& master_node_addr,
    int32_t dp_size) {
  const auto& devices = options.devices();
  // The allocator, physical page pool and global mapping currently share one
  // process-wide device context. Reject multiple devices before starting RPCs.
  CHECK_EQ(devices.size(), 1) << "Worker memory RPC requires exactly "
                                 "one local device per process";
  CHECK_GT(dp_size, 0) << "dp_size must be positive";
  CHECK(servers_.empty()) << "Worker memory RPC already set up";

  const auto& distributed_config = DistributedConfig::get_instance();
  CHECK_GT(distributed_config.nnodes(), 0) << "nnodes must be positive";
  CHECK_GE(distributed_config.node_rank(), 0);
  CHECK_LT(distributed_config.node_rank(), distributed_config.nnodes());
  const int32_t world_size = distributed_config.nnodes();
  CHECK_EQ(world_size % dp_size, 0)
      << "world_size must be divisible by dp_size";

  const int32_t tp_size = world_size / dp_size;
  configure_workers(world_size, dp_size, tp_size);

  std::shared_ptr<CollectiveService> collective_service;
  if (distributed_config.node_rank() == 0) {
    collective_service = std::make_shared<CollectiveService>(world_size);
    XllmServer* collective_server =
        ServerRegistry::get_instance().register_server(collective_server_name_);
    CHECK(collective_server->start(
        collective_service, master_node_addr, collective_server_name_))
        << "Failed to start worker memory collective server on address: "
        << master_node_addr;
  }

  servers_.emplace_back(std::make_unique<WorkerMemoryRpcServer>(
      /*local_rank=*/0, master_node_addr, devices.front(), options));
  CHECK(servers_.front()->wait_until_ready())
      << "Failed to start or register local worker memory RPC server";

  if (collective_service == nullptr) {
    return;
  }
  auto worker_memory_rpc_addrs = collective_service->wait();

  auto& clients = clients_;
  clients.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    auto address = worker_memory_rpc_addrs.find(rank);
    CHECK(address != worker_memory_rpc_addrs.end())
        << "worker memory RPC server did not connect to master node: rank "
        << rank;
    auto client = std::make_shared<WorkerMemoryRpcClient>(
        rank, address->second, devices.front());
    clients.emplace_back(client);
  }

  LOG(INFO) << "Worker memory RPC setup: world_size=" << world_size
            << ", dp_size=" << dp_size << ", tp_size=" << tp_size;
}

int64_t DistributedMemoryCoordinator::init_physical_page_pools(
    double max_memory_utilization,
    int64_t max_cache_size) {
  if (world_size_ <= 1) {
    // Single process single GPU, initialize locally
    auto& resources = WorkerMemoryResources::get_instance();
    const torch::Device& local_device = resources.device();
    Device device(local_device);
    device.set_device();

    const int64_t available_memory = device.free_memory();
    const int64_t total_memory = device.total_memory();

    int64_t cache_size = available_memory;
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          static_cast<int64_t>(total_memory * (1.0 - max_memory_utilization));
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

    resources.init_physical_page_pool(num_pages);
    LOG(INFO) << "GlobalMemoryRegion initialized (local)";

    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(clients_.size());
  for (auto& client : clients_) {
    memory_futures.emplace_back(client->get_memory_info_async());
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
        static_cast<int64_t>(min_total_memory * (1.0 - max_memory_utilization));
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
  init_futures.reserve(clients_.size());
  for (auto& client : clients_) {
    init_futures.emplace_back(client->init_physical_page_pool_async(num_pages));
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
            << world_size_ << " workers with " << num_pages << " pages each";
  return num_pages;
}

void DistributedMemoryCoordinator::set_model_parallel_strategy(
    const std::string& model_id,
    int32_t dp_size,
    int32_t tp_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_GT(dp_size, 0);
  CHECK_GT(tp_size, 0);
  CHECK_LE(static_cast<int64_t>(dp_size) * tp_size,
           budget_initialized_ ? page_budget_.worker_count() : world_size_);
  model_parallel_strategies_[model_id] = {dp_size, tp_size};
  LOG(INFO) << "Set model parallel strategy for " << model_id
            << ": dp_size=" << dp_size << ", tp_size=" << tp_size;
}

std::pair<int32_t, int32_t>
DistributedMemoryCoordinator::get_model_parallel_strategy(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return parallel_strategy_locked(model_id);
}

std::pair<int32_t, int32_t>
DistributedMemoryCoordinator::parallel_strategy_locked(
    const std::string& model_id) const {
  const auto it = model_parallel_strategies_.find(model_id);
  return it == model_parallel_strategies_.end()
             ? std::make_pair(dp_size_, tp_size_)
             : it->second;
}

bool DistributedMemoryCoordinator::broadcast_map_to_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just map locally
    return WorkerMemoryResources::get_instance().map_to_kv_tensors(model_id,
                                                                   offsets);
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
       r < end_rank && r < static_cast<int32_t>(clients_.size());
       ++r) {
    futures.emplace_back(
        clients_[r]->map_to_kv_tensors_async(model_id, offsets));
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

bool DistributedMemoryCoordinator::broadcast_unmap_from_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  if (world_size_ <= 1) {
    // Single process single GPU, just unmap locally
    return WorkerMemoryResources::get_instance().unmap_from_kv_tensors(model_id,
                                                                       offsets);
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
       r < end_rank && r < static_cast<int32_t>(clients_.size());
       ++r) {
    futures.emplace_back(
        clients_[r]->unmap_from_kv_tensors_async(model_id, offsets));
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

bool DistributedMemoryCoordinator::broadcast_alloc_weight_pages(
    const std::string& model_id,
    size_t num_pages) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (model_world_size <= 1) {
    return WorkerMemoryResources::get_instance().alloc_weight_pages(model_id,
                                                                    num_pages);
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers =
      std::min(model_world_size, static_cast<int32_t>(clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.emplace_back(
        clients_[i]->alloc_weight_pages_async(model_id, num_pages));
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

bool DistributedMemoryCoordinator::broadcast_free_weight_pages(
    const std::string& model_id) {
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (model_world_size <= 1) {
    // Single process: free locally
    WorkerMemoryResources::get_instance().free_weight(model_id);
    return true;
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  int32_t num_workers =
      std::min(model_world_size, static_cast<int32_t>(clients_.size()));
  futures.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    futures.emplace_back(clients_[i]->free_weight_pages_async(model_id));
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

bool DistributedMemoryCoordinator::get_kv_cache_offsets(
    int32_t dp_rank,
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    uint64_t block_size_bytes,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) {
  const auto [model_dp_size, model_tp_size] =
      get_model_parallel_strategy(model_id);
  if (dp_rank < 0 || dp_rank >= model_dp_size) {
    LOG(ERROR) << "Invalid DP rank " << dp_rank << " for model " << model_id;
    return false;
  }
  // Cache mappings in the master's DP group have identical pool-relative
  // offsets and can be queried locally.
  if (dp_rank == 0) {
    const std::optional<int64_t> model_num_layers =
        WorkerMemoryResources::get_instance().get_kv_cache_num_layers(model_id);
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
        auto [k_offset, v_offset] =
            WorkerMemoryResources::get_instance().get_global_offsets_for_block(
                model_id, layer_id, block_id, block_size_bytes);
        if (k_offset == UINT64_MAX || v_offset == UINT64_MAX) {
          LOG(ERROR) << "Failed to get local offsets for block " << block_id
                     << " at layer " << layer_id;
          return false;
        }
        k_offsets.emplace_back(k_offset);
        v_offsets.emplace_back(v_offset);
      }
      layer_offsets[layer_id] = {std::move(k_offsets), std::move(v_offsets)};
    }

    VLOG(1) << "get_kv_cache_offsets (local): model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
    return true;
  }

  // Forked models can use a different TP size from the initial cluster.
  // Query the first worker in this model's DP group, matching map/unmap
  // routing.
  const size_t worker_rank = static_cast<size_t>(dp_rank * model_tp_size);
  if (worker_rank >= clients_.size()) {
    LOG(ERROR) << "No worker memory RPC client for rank " << worker_rank;
    return false;
  }
  const auto& client = clients_[worker_rank];
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

void DistributedMemoryCoordinator::init_page_budget(size_t total_pages,
                                                    int32_t dp_size,
                                                    int32_t worker_count,
                                                    size_t page_size,
                                                    bool enable_prealloc) {
  std::lock_guard<std::mutex> initialization_lock(initialization_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (budget_initialized_) {
      return;
    }
    CHECK_GT(dp_size, 0);
    CHECK_GT(worker_count, 0);
    CHECK_EQ(worker_count % dp_size, 0);
    CHECK_GT(page_size, 0);
    dp_size_ = dp_size;
    tp_size_ = worker_count / dp_size;
    page_budget_.reset(total_pages, worker_count);
  }
  // The KV allocator can enter the backend while holding its own lock.
  kv_cache_pages_->init(total_pages, page_size, dp_size, enable_prealloc);
  std::lock_guard<std::mutex> lock(mutex_);
  budget_initialized_ = true;
}

bool DistributedMemoryCoordinator::is_initialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return budget_initialized_;
}

size_t DistributedMemoryCoordinator::get_num_total_phy_pages() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.total_pages();
}

PagedKVCachePageAllocator&
DistributedMemoryCoordinator::kv_cache_page_allocator() {
  return *kv_cache_pages_;
}

int32_t DistributedMemoryCoordinator::model_worker_count_locked(
    const std::string& model_id) const {
  const auto [dp_size, tp_size] = parallel_strategy_locked(model_id);
  return std::min(dp_size * tp_size, page_budget_.worker_count());
}

std::pair<int32_t, int32_t>
DistributedMemoryCoordinator::dp_worker_range_locked(
    const std::string& model_id,
    int32_t dp_rank) const {
  const auto [dp_size, tp_size] = parallel_strategy_locked(model_id);
  if (dp_rank < 0 || dp_rank >= dp_size) {
    return {0, 0};
  }
  const int32_t first = dp_rank * tp_size;
  return {first, std::min(first + tp_size, page_budget_.worker_count())};
}

bool DistributedMemoryCoordinator::try_reserve_pages(
    const std::string& model_id,
    int32_t dp_rank,
    size_t physical_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(budget_initialized_);
  const auto [first, last] = dp_worker_range_locked(model_id, dp_rank);
  if (first == last) {
    return false;
  }
  const bool reserved = page_budget_.try_reserve(first, last, physical_pages);
  update_memory_usage_locked();
  return reserved;
}

void DistributedMemoryCoordinator::release_pages(const std::string& model_id,
                                                 int32_t dp_rank,
                                                 size_t physical_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(budget_initialized_);
  const auto [first, last] = dp_worker_range_locked(model_id, dp_rank);
  CHECK_LT(first, last);
  page_budget_.release(first, last, physical_pages);
  update_memory_usage_locked();
}

size_t DistributedMemoryCoordinator::available_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [first, last] = dp_worker_range_locked(model_id, dp_rank);
  return first == last ? 0 : page_budget_.min_free_pages(first, last);
}

bool DistributedMemoryCoordinator::map_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  return broadcast_map_to_kv_tensors(model_id, dp_rank, offsets);
}

bool DistributedMemoryCoordinator::unmap_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  return broadcast_unmap_from_kv_tensors(model_id, dp_rank, offsets);
}

bool DistributedMemoryCoordinator::reserve_weight_pages(
    const std::string& model_id,
    size_t num_pages) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(budget_initialized_);
    if (!page_budget_.try_reserve(
            0, model_worker_count_locked(model_id), num_pages)) {
      return false;
    }
    weight_page_counts_[model_id] = num_pages;
    update_memory_usage_locked();
  }
  if (num_pages > 0) {
    CHECK(broadcast_alloc_weight_pages(model_id, num_pages))
        << "Weight allocation failed; distributed mapping state is uncertain";
  }
  return true;
}

void DistributedMemoryCoordinator::set_weight_page_count(
    const std::string& model_id,
    size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  weight_page_counts_[model_id] = num_pages;
}

size_t DistributedMemoryCoordinator::get_weight_page_count(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = weight_page_counts_.find(model_id);
  return it == weight_page_counts_.end() ? 0 : it->second;
}

void DistributedMemoryCoordinator::release_weight_pages(
    const std::string& model_id) {
  const size_t pages = get_weight_page_count(model_id);
  if (pages == 0) {
    return;
  }
  CHECK(broadcast_free_weight_pages(model_id))
      << "Weight release failed; distributed mapping state is uncertain";
  std::lock_guard<std::mutex> lock(mutex_);
  page_budget_.release(0, model_worker_count_locked(model_id), pages);
  update_memory_usage_locked();
}

void DistributedMemoryCoordinator::restore_weight_pages(
    const std::string& model_id) {
  const size_t pages = get_weight_page_count(model_id);
  if (pages > 0) {
    CHECK(broadcast_alloc_weight_pages(model_id, pages))
        << "Weight restore failed; distributed mapping state is uncertain";
  }
}

bool DistributedMemoryCoordinator::try_reserve_model_memory(
    const std::string& model_id,
    const KVCacheMappingPlan& plan) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(budget_initialized_);
  std::vector<size_t> demand(page_budget_.worker_count(), 0);
  for (size_t rank = 0; rank < plan.dp_group_page_ids.size(); ++rank) {
    const auto& ids = plan.dp_group_page_ids[rank];
    if (ids.empty()) {
      continue;
    }
    const auto [first, last] =
        dp_worker_range_locked(model_id, static_cast<int32_t>(rank));
    CHECK_LT(first, last);
    const size_t pages = ids.size() * plan.physical_pages_per_virtual_page;
    for (int32_t worker = first; worker < last; ++worker) {
      demand[worker] += pages;
    }
  }
  const auto weight = weight_page_counts_.find(model_id);
  const size_t weight_pages =
      weight == weight_page_counts_.end() ? 0 : weight->second;
  const int32_t worker_count = model_worker_count_locked(model_id);
  for (int32_t worker = 0; worker < worker_count; ++worker) {
    demand[worker] += weight_pages;
  }
  const bool reserved = page_budget_.try_reserve(demand);
  update_memory_usage_locked();
  return reserved;
}

void DistributedMemoryCoordinator::release_kv_mapping_budget(
    const std::string& model_id,
    const KVCacheMappingPlan& plan) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t rank = 0; rank < plan.dp_group_page_ids.size(); ++rank) {
    const auto& ids = plan.dp_group_page_ids[rank];
    if (ids.empty()) {
      continue;
    }
    const auto [first, last] =
        dp_worker_range_locked(model_id, static_cast<int32_t>(rank));
    CHECK_LT(first, last);
    page_budget_.release(
        first, last, ids.size() * plan.physical_pages_per_virtual_page);
  }
  update_memory_usage_locked();
}

std::vector<size_t> DistributedMemoryCoordinator::get_worker_free_page_counts()
    const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.free_pages_snapshot();
}

size_t DistributedMemoryCoordinator::get_free_pages_for_model(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.min_free_pages(0, model_worker_count_locked(model_id));
}

void DistributedMemoryCoordinator::update_memory_usage_locked() const {
  VLOG(2) << "Shared worker budget: total_pages=" << page_budget_.total_pages()
          << ", min_free_pages="
          << page_budget_.min_free_pages(0, page_budget_.worker_count());
}

}  // namespace xllm

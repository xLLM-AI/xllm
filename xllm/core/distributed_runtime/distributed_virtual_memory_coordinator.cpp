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

#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "core/distributed_runtime/collective_service.h"
#include "core/distributed_runtime/worker_memory_rpc_client.h"
#include "core/distributed_runtime/worker_memory_rpc_server.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"
#include "core/platform/device.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_memory_resources.h"
#include "server/xllm_server_registry.h"

namespace xllm {

namespace {

constexpr int32_t kCapacityQueryTimeoutMs = 1000;
constexpr std::chrono::milliseconds kCapacityRefreshInterval(250);

}  // namespace

DistributedVirtualMemoryCoordinator::DistributedVirtualMemoryCoordinator() {
  // These singletons must outlive the coordinator and its RPC shutdown.
  WorkerMemoryResources::get_instance();
  ServerRegistry::get_instance();
  kv_cache_pages_ = std::make_unique<VmmKVPageAllocatorImpl>(*this);
}

DistributedVirtualMemoryCoordinator::~DistributedVirtualMemoryCoordinator() {
  destroy();
}

void DistributedVirtualMemoryCoordinator::prepare_kv_cache_pool(
    const runtime::Options& options,
    int64_t world_size,
    size_t worker_count,
    int64_t reserved_state_bytes,
    bool is_multimodal) {
  CHECK(reserved_state_bytes == 0 ||
        !LoadConfig::get_instance().enable_vmm_weights())
      << "Virtual weight memory does not support hybrid linear-state caches.";
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (!phy_pool.is_initialized()) {
    const int64_t encoder_cache_reserved_bytes =
        is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024 : 0;
    CHECK_GT(
        init_physical_page_pools(config.max_memory_utilization(),
                                 config.max_cache_size(),
                                 config.max_memory_utilization_configured(),
                                 encoder_cache_reserved_bytes,
                                 reserved_state_bytes),
        0)
        << "Failed to initialize KV cache physical page pools";
  }
  if (!is_initialized()) {
    init_page_budget(phy_pool.num_total(),
                     options.dp_size(),
                     static_cast<int32_t>(worker_count),
                     config.phy_page_granularity_size(),
                     /*enable_prealloc=*/true);
  }
  set_model_parallel_strategy(
      options.model_id(), options.dp_size(), static_cast<int32_t>(world_size));
}

void DistributedVirtualMemoryCoordinator::destroy() {
  stop_worker_capacity_refresh();
  // Join executing maps and cancel queued preallocation while the backend and
  // its RPCs are still alive. Cache users must already have stopped; this is
  // whole-process teardown, not an online model unload operation.
  if (kv_cache_pages_) {
    kv_cache_pages_->shutdown();
  }
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
  weight_page_counts_.clear();
  model_parallel_strategies_.clear();
  clients_.clear();
  worker_allocatable_pages_.clear();
  budget_initialized_ = false;
  world_size_ = 0;
  node_rank_ = 0;
  dp_size_ = 1;
  tp_size_ = 1;
  auto& resources = WorkerMemoryResources::get_instance();
  if (resources.is_initialized()) {
    resources.destroy();
  }
}

void DistributedVirtualMemoryCoordinator::configure_workers(int32_t world_size,
                                                            int32_t dp_size,
                                                            int32_t tp_size) {
  world_size_ = world_size;
  dp_size_ = dp_size;
  tp_size_ = tp_size;
  clients_.clear();
  clients_.reserve(world_size_);
  worker_allocatable_pages_.clear();
  worker_allocatable_pages_.resize(static_cast<size_t>(world_size_), 0);
}

void DistributedVirtualMemoryCoordinator::setup_worker_memory_rpc(
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
  node_rank_ = distributed_config.node_rank();

  const auto timeout = std::chrono::milliseconds(
      KVCacheConfig::get_instance().vmm_startup_timeout_ms());
  // Start the registration deadline before workers can register and begin
  // their own pool-readiness waits.
  const auto registration_deadline = std::chrono::steady_clock::now() + timeout;
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
  // Bound the master's registration stage with the same configured timeout
  // used by non-master nodes waiting for pool initialization.
  auto worker_memory_rpc_addrs =
      collective_service->wait_until(registration_deadline);
  CHECK(worker_memory_rpc_addrs.has_value())
      << "Timed out waiting for worker memory RPC registration: world_size="
      << world_size << ", timeout_ms=" << timeout.count();

  auto& clients = clients_;
  clients.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    auto address = worker_memory_rpc_addrs->find(rank);
    CHECK(address != worker_memory_rpc_addrs->end())
        << "worker memory RPC server did not connect to master node: rank "
        << rank;
    auto client = std::make_shared<WorkerMemoryRpcClient>(
        rank, address->second, devices.front());
    clients.emplace_back(client);
  }

  LOG(INFO) << "Worker memory RPC setup: world_size=" << world_size
            << ", dp_size=" << dp_size << ", tp_size=" << tp_size;
}

int64_t DistributedVirtualMemoryCoordinator::init_physical_page_pools(
    double max_memory_utilization,
    int64_t max_cache_size,
    bool max_memory_utilization_configured,
    int64_t reserved_memory_bytes,
    int64_t resident_memory_bytes) {
  CHECK(std::isfinite(max_memory_utilization));
  CHECK_GE(max_memory_utilization, 0.0);
  CHECK_LE(max_memory_utilization, 1.0);
  CHECK_GE(max_cache_size, 0);
  CHECK_GE(reserved_memory_bytes, 0);
  const int64_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  CHECK_GT(page_size, 0);
  const bool lazy = KVCacheConfig::get_instance().vmm_elastic();
  if (world_size_ <= 1) {
    // Single process single GPU, initialize locally
    auto& resources = WorkerMemoryResources::get_instance();
    const torch::Device& local_device = resources.device();
    Device device(local_device);
    device.set_device();

    const int64_t available_memory = device.free_memory();
    const int64_t total_memory = device.total_memory();
    if (available_memory < 0 || total_memory <= 0 ||
        available_memory > total_memory) {
      LOG(ERROR) << "Local worker returned invalid memory info";
      return 0;
    }

    const VmmArenaCapacity capacity =
        calculate_vmm_arena_capacity(available_memory,
                                     total_memory,
                                     max_memory_utilization,
                                     max_memory_utilization_configured,
                                     lazy,
                                     max_cache_size,
                                     reserved_memory_bytes,
                                     resident_memory_bytes);

    const int64_t num_pages = capacity.maximum_bytes / page_size;
    const int64_t initial_pages = capacity.initial_bytes / page_size;
    LOG(INFO) << "init_physical_page_pools (local): available_memory="
              << available_memory << ", total_memory=" << total_memory
              << ", cache_size=" << capacity.maximum_bytes
              << ", num_pages=" << num_pages
              << ", initial_pages=" << initial_pages;

    if (num_pages <= 0) {
      LOG(ERROR) << "Insufficient memory for physical page address capacity";
      return 0;
    }
    // Elastic pools reserve the maximum stable address arena without
    // allocating backing pages; mapping checks live device pressure later.
    resources.init_physical_page_pool(
        num_pages,
        initial_pages,
        lazy,
        static_cast<size_t>(capacity.growth_memory_reserve_bytes));
    LOG(INFO) << "GlobalVirtualMemoryRegion initialized (local)";

    {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_allocatable_pages_.assign(1, static_cast<size_t>(num_pages));
    }
    refresh_worker_capacity_snapshot();
    start_worker_capacity_refresh();
    return num_pages;
  }

  if (node_rank_ != 0) {
    // Only the master owns clients for all workers. Other nodes keep their
    // local server running until the master's initialization RPC completes.
    // Registration can outlast a single RPC. Share the master's configured
    // per-stage startup timeout instead of using the per-request RPC timeout.
    const auto timeout = std::chrono::milliseconds(
        KVCacheConfig::get_instance().vmm_startup_timeout_ms());
    const int64_t num_pages =
        WorkerMemoryResources::get_instance().wait_for_physical_page_pool(
            timeout);
    if (num_pages <= 0) {
      LOG(ERROR) << "Local physical page pool was not initialized by master: "
                 << "node_rank=" << node_rank_
                 << ", timeout_ms=" << timeout.count();
    }
    return num_pages;
  }

  // Step 1: Query available memory from all workers via RPC
  if (clients_.size() != static_cast<size_t>(world_size_)) {
    LOG(ERROR) << "Worker memory RPC client count does not match world size";
    return 0;
  }
  std::vector<folly::SemiFuture<MemoryInfo>> memory_futures;
  memory_futures.reserve(clients_.size());
  for (auto& client : clients_) {
    memory_futures.emplace_back(client->get_memory_info_async());
  }

  // Wait for all memory info responses
  auto memory_results = folly::collectAll(memory_futures).get();

  int64_t min_available_memory = std::numeric_limits<int64_t>::max();
  int64_t min_total_memory = std::numeric_limits<int64_t>::max();
  int64_t min_capacity_bytes = std::numeric_limits<int64_t>::max();
  int64_t min_initial_bytes = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> growth_memory_reserves;
  growth_memory_reserves.reserve(memory_results.size());

  for (size_t i = 0; i < memory_results.size(); ++i) {
    if (!memory_results[i].hasValue()) {
      LOG(ERROR) << "Failed to get memory info from worker: " << i;
      return 0;
    }
    auto& info = memory_results[i].value();
    if (info.available_memory < 0 || info.total_memory <= 0 ||
        info.available_memory > info.total_memory) {
      LOG(ERROR) << "Worker " << i << " returned invalid memory info";
      return 0;
    }

    LOG(INFO) << "Worker #" << i
              << ": available_memory=" << info.available_memory
              << ", total_memory=" << info.total_memory;

    min_available_memory =
        std::min(min_available_memory, info.available_memory);
    min_total_memory = std::min(min_total_memory, info.total_memory);
    const VmmArenaCapacity capacity =
        calculate_vmm_arena_capacity(info.available_memory,
                                     info.total_memory,
                                     max_memory_utilization,
                                     max_memory_utilization_configured,
                                     lazy,
                                     max_cache_size,
                                     reserved_memory_bytes,
                                     resident_memory_bytes);
    // Apply each worker's ratio before reducing so heterogeneous devices keep
    // their own configured arena ceiling.
    min_capacity_bytes = std::min(min_capacity_bytes, capacity.maximum_bytes);
    min_initial_bytes = std::min(min_initial_bytes, capacity.initial_bytes);
    growth_memory_reserves.emplace_back(capacity.growth_memory_reserve_bytes);
  }

  // Step 2: Convert the smallest worker capacity to the stable arena size.
  const int64_t cache_size = min_capacity_bytes;

  const int64_t num_pages = cache_size / page_size;
  const int64_t initial_pages = min_initial_bytes / page_size;
  LOG(INFO) << "init_physical_page_pools: min_available_memory="
            << min_available_memory << ", min_total_memory=" << min_total_memory
            << ", cache_size=" << cache_size << ", num_pages=" << num_pages
            << ", initial_pages=" << initial_pages;

  if (num_pages <= 0) {
    LOG(ERROR) << "Insufficient memory for PhysicalPagePool";
    return 0;
  }

  // Step 3: Broadcast InitPhysicalPagePool to all workers
  std::vector<folly::SemiFuture<bool>> init_futures;
  init_futures.reserve(clients_.size());
  for (size_t worker = 0; worker < clients_.size(); ++worker) {
    init_futures.emplace_back(clients_[worker]->init_physical_page_pool_async(
        num_pages,
        initial_pages,
        lazy,
        static_cast<size_t>(growth_memory_reserves[worker])));
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_allocatable_pages_.assign(static_cast<size_t>(world_size_),
                                     static_cast<size_t>(num_pages));
  }
  refresh_worker_capacity_snapshot();
  start_worker_capacity_refresh();
  return num_pages;
}

void DistributedVirtualMemoryCoordinator::set_model_parallel_strategy(
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
DistributedVirtualMemoryCoordinator::get_model_parallel_strategy(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return parallel_strategy_locked(model_id);
}

std::pair<int32_t, int32_t>
DistributedVirtualMemoryCoordinator::parallel_strategy_locked(
    const std::string& model_id) const {
  const auto it = model_parallel_strategies_.find(model_id);
  return it == model_parallel_strategies_.end()
             ? std::make_pair(dp_size_, tp_size_)
             : it->second;
}

bool DistributedVirtualMemoryCoordinator::broadcast_map_to_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  std::lock_guard<std::mutex> transaction_lock(mapping_transaction_mutex_);
  CHECK_NE(next_mapping_transaction_id_, std::numeric_limits<uint64_t>::max());
  const uint64_t transaction_id = next_mapping_transaction_id_++;
  if (world_size_ <= 1) {
    auto& resources = WorkerMemoryResources::get_instance();
    if (resources.prepare_kv_mapping(transaction_id, model_id, offsets) &&
        resources.commit_kv_mapping(transaction_id)) {
      return true;
    }
    CHECK(resources.abort_kv_mapping(transaction_id))
        << "Local KV mapping rollback failed; physical ownership is uncertain";
    return false;
  }

  const auto [model_dp_size, model_tp_size] =
      get_model_parallel_strategy(model_id);
  if (dp_rank < 0 || dp_rank >= model_dp_size) {
    return false;
  }
  const int32_t start_rank = dp_rank * model_tp_size;
  const int32_t end_rank = start_rank + model_tp_size;
  if (end_rank > static_cast<int32_t>(clients_.size())) {
    return false;
  }
  const auto all_succeeded = [](std::vector<folly::SemiFuture<bool>>& futures) {
    const auto results = folly::collectAll(futures).get();
    return std::all_of(results.begin(), results.end(), [](const auto& result) {
      return result.hasValue() && result.value();
    });
  };

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_tp_size);
  for (int32_t worker = start_rank; worker < end_rank; ++worker) {
    futures.emplace_back(clients_[worker]->prepare_kv_mapping_async(
        transaction_id, model_id, offsets));
  }
  bool succeeded = all_succeeded(futures);
  if (succeeded) {
    futures.clear();
    for (int32_t worker = start_rank; worker < end_rank; ++worker) {
      futures.emplace_back(
          clients_[worker]->commit_kv_mapping_async(transaction_id));
    }
    succeeded = all_succeeded(futures);
  }
  if (succeeded) {
    return true;
  }

  // An RPC failure does not prove the worker did nothing. Abort every
  // participant, including workers whose prepare/commit reply was lost.
  futures.clear();
  for (int32_t worker = start_rank; worker < end_rank; ++worker) {
    futures.emplace_back(
        clients_[worker]->abort_kv_mapping_async(transaction_id));
  }
  CHECK(all_succeeded(futures)) << "Distributed KV mapping rollback was not "
                                   "acknowledged by every worker; "
                                   "physical page ownership is uncertain";
  return false;
}

bool DistributedVirtualMemoryCoordinator::broadcast_unmap_from_kv_tensors(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  std::lock_guard<std::mutex> transaction_lock(mapping_transaction_mutex_);
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
  if (end_rank > static_cast<int32_t>(clients_.size())) {
    return false;
  }

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

bool DistributedVirtualMemoryCoordinator::broadcast_alloc_weight_pages(
    const std::string& model_id,
    size_t num_pages) {
  std::lock_guard<std::mutex> transaction_lock(mapping_transaction_mutex_);
  CHECK_NE(next_mapping_transaction_id_, std::numeric_limits<uint64_t>::max());
  const uint64_t transaction_id = next_mapping_transaction_id_++;
  const auto [model_dp_size, model_tp_size] =
      get_model_parallel_strategy(model_id);
  const int32_t model_world_size = model_dp_size * model_tp_size;
  if (world_size_ <= 1) {
    auto& resources = WorkerMemoryResources::get_instance();
    if (resources.prepare_weight_pages(transaction_id, model_id, num_pages) &&
        resources.commit_weight_pages(transaction_id)) {
      return true;
    }
    CHECK(resources.abort_weight_pages(transaction_id))
        << "Weight rollback failed; physical ownership is uncertain";
    return false;
  }
  if (model_world_size > static_cast<int32_t>(clients_.size())) {
    return false;
  }
  const auto all_succeeded = [](std::vector<folly::SemiFuture<bool>>& futures) {
    const auto results = folly::collectAll(futures).get();
    return std::all_of(results.begin(), results.end(), [](const auto& result) {
      return result.hasValue() && result.value();
    });
  };
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(model_world_size);
  for (int32_t worker = 0; worker < model_world_size; ++worker) {
    futures.emplace_back(clients_[worker]->prepare_weight_pages_async(
        transaction_id, model_id, num_pages));
  }
  bool succeeded = all_succeeded(futures);
  if (succeeded) {
    futures.clear();
    for (int32_t worker = 0; worker < model_world_size; ++worker) {
      futures.emplace_back(
          clients_[worker]->commit_weight_pages_async(transaction_id));
    }
    succeeded = all_succeeded(futures);
  }
  if (succeeded) {
    return true;
  }
  // Roll back every target, including workers whose result is unknown. The
  // ledger remains reserved until every worker confirms physical cleanup.
  futures.clear();
  for (int32_t worker = 0; worker < model_world_size; ++worker) {
    futures.emplace_back(
        clients_[worker]->abort_weight_pages_async(transaction_id));
  }
  CHECK(all_succeeded(futures))
      << "Distributed weight rollback was not acknowledged by every worker; "
         "physical page ownership is uncertain";
  return false;
}

bool DistributedVirtualMemoryCoordinator::broadcast_free_weight_pages(
    const std::string& model_id) {
  std::lock_guard<std::mutex> transaction_lock(mapping_transaction_mutex_);
  // Get model-specific parallel strategy
  auto [model_dp_size, model_tp_size] = get_model_parallel_strategy(model_id);
  int32_t model_world_size = model_dp_size * model_tp_size;

  if (world_size_ <= 1) {
    // Single process: free locally
    WorkerMemoryResources::get_instance().free_weight(model_id);
    return true;
  }

  // Broadcast to all workers for this model
  std::vector<folly::SemiFuture<bool>> futures;
  if (model_world_size > static_cast<int32_t>(clients_.size())) {
    return false;
  }
  const int32_t num_workers = model_world_size;
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

bool DistributedVirtualMemoryCoordinator::get_virtual_kv_cache_offsets(
    int32_t dp_rank,
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    std::vector<VirtualKVCacheLayerGlobalOffsets>& layer_offsets) {
  layer_offsets.clear();
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
    std::vector<VirtualKVCacheLayerGlobalOffsets> local_layer_offsets(
        num_layers);

    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      auto& layer = local_layer_offsets[layer_id];
      layer.k_offsets.reserve(block_ids.size());
      layer.v_offsets.reserve(block_ids.size());
      layer.index_offsets.reserve(block_ids.size());

      for (int32_t block_id : block_ids) {
        const auto offsets =
            WorkerMemoryResources::get_instance().get_global_offsets_for_block(
                model_id, layer_id, block_id);
        if (!offsets.has_value()) {
          LOG(ERROR) << "Failed to get local offsets for block " << block_id
                     << " at layer " << layer_id;
          return false;
        }
        layer.k_offsets.emplace_back(offsets->k_offset);
        layer.v_offsets.emplace_back(offsets->v_offset);
        if (offsets->index_offset.has_value()) {
          layer.index_offsets.emplace_back(offsets->index_offset.value());
        }
      }
    }
    layer_offsets = std::move(local_layer_offsets);

    VLOG(1) << "get_virtual_kv_cache_offsets (local): model_id=" << model_id
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
  layer_offsets = client->get_virtual_kv_cache_offsets(model_id, block_ids);
  if (layer_offsets.empty()) {
    LOG(ERROR) << "get_virtual_kv_cache_offsets failed for dp_rank=" << dp_rank
               << ", model_id=" << model_id;
    return false;
  }

  VLOG(1) << "get_virtual_kv_cache_offsets: dp_rank=" << dp_rank
          << ", model_id=" << model_id << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();

  return true;
}

void DistributedVirtualMemoryCoordinator::init_page_budget(
    size_t total_pages,
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
  const bool elastic = KVCacheConfig::get_instance().vmm_elastic();
  kv_cache_pages_->init(total_pages,
                        page_size,
                        dp_size,
                        enable_prealloc && !elastic,
                        /*max_reserved_pages=*/elastic ? 0 : 32);
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_allocatable_pages_.size() != static_cast<size_t>(worker_count)) {
    worker_allocatable_pages_.assign(worker_count, total_pages);
  }
  budget_initialized_ = true;
}

bool DistributedVirtualMemoryCoordinator::is_initialized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return budget_initialized_;
}

size_t DistributedVirtualMemoryCoordinator::get_num_total_phy_pages() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.total_pages();
}

VmmKVPageAllocatorImpl&
DistributedVirtualMemoryCoordinator::vmm_kv_page_allocator() {
  return *kv_cache_pages_;
}

int32_t DistributedVirtualMemoryCoordinator::model_worker_count_locked(
    const std::string& model_id) const {
  const auto [dp_size, tp_size] = parallel_strategy_locked(model_id);
  return std::min(dp_size * tp_size, page_budget_.worker_count());
}

std::pair<int32_t, int32_t>
DistributedVirtualMemoryCoordinator::dp_worker_range_locked(
    const std::string& model_id,
    int32_t dp_rank) const {
  const auto [dp_size, tp_size] = parallel_strategy_locked(model_id);
  if (dp_rank < 0 || dp_rank >= dp_size) {
    return {0, 0};
  }
  const int32_t first = dp_rank * tp_size;
  return {first, std::min(first + tp_size, page_budget_.worker_count())};
}

bool DistributedVirtualMemoryCoordinator::try_reserve_pages(
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

void DistributedVirtualMemoryCoordinator::release_pages(
    const std::string& model_id,
    int32_t dp_rank,
    size_t physical_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(budget_initialized_);
  const auto [first, last] = dp_worker_range_locked(model_id, dp_rank);
  CHECK_LT(first, last);
  page_budget_.release(first, last, physical_pages);
  update_memory_usage_locked();
}

size_t DistributedVirtualMemoryCoordinator::available_pages(
    const std::string& model_id,
    int32_t dp_rank) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [first, last] = dp_worker_range_locked(model_id, dp_rank);
  if (first == last) {
    return 0;
  }
  size_t available = page_budget_.min_free_pages(first, last);
  if (!KVCacheConfig::get_instance().vmm_elastic()) {
    return available;
  }
  if (last > static_cast<int32_t>(worker_allocatable_pages_.size())) {
    return 0;
  }
  for (int32_t worker = first; worker < last; ++worker) {
    available = std::min(
        available, worker_allocatable_pages_[static_cast<size_t>(worker)]);
  }
  // This is an advisory snapshot. Mapping prepare rechecks the live worker
  // budget before publishing any page ownership.
  return available;
}

void DistributedVirtualMemoryCoordinator::refresh_worker_capacity_snapshot() {
  const auto invalidate_snapshot = [this]() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(
        worker_allocatable_pages_.begin(), worker_allocatable_pages_.end(), 0);
  };

  std::vector<std::shared_ptr<WorkerMemoryRpcClient>> workers;
  bool distributed = false;
  int32_t expected_worker_count = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    distributed = world_size_ > 1;
    expected_worker_count = world_size_;
    if (distributed) {
      workers = clients_;
    }
  }

  std::vector<size_t> capacities;
  capacities.reserve(distributed ? workers.size() : 1);
  if (!distributed) {
    const WorkerMemoryCapacity capacity =
        WorkerMemoryResources::get_instance().memory_capacity_snapshot();
    if (!capacity.pool_initialized) {
      invalidate_snapshot();
      return;
    }
    capacities.emplace_back(capacity.pool_allocatable_pages);
  } else {
    if (workers.size() != static_cast<size_t>(expected_worker_count)) {
      invalidate_snapshot();
      return;
    }
    std::vector<folly::SemiFuture<MemoryInfo>> futures;
    futures.reserve(workers.size());
    for (const auto& worker : workers) {
      futures.emplace_back(
          worker->get_memory_info_async(kCapacityQueryTimeoutMs));
    }
    const auto results = folly::collectAll(futures).get();
    capacities.reserve(results.size());
    for (const auto& result : results) {
      if (!result.hasValue() || !result.value().pool_initialized) {
        invalidate_snapshot();
        return;
      }
      capacities.emplace_back(result.value().pool_allocatable_pages);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (capacities.size() == worker_allocatable_pages_.size()) {
    worker_allocatable_pages_ = std::move(capacities);
  } else {
    std::fill(
        worker_allocatable_pages_.begin(), worker_allocatable_pages_.end(), 0);
  }
}

void DistributedVirtualMemoryCoordinator::start_worker_capacity_refresh() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (capacity_refresh_running_ ||
      !KVCacheConfig::get_instance().vmm_elastic()) {
    return;
  }
  capacity_refresh_running_ = true;
  capacity_refresh_thread_ = std::thread([this] {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!capacity_refresh_running_) {
          return;
        }
      }
      refresh_worker_capacity_snapshot();
      std::unique_lock<std::mutex> lock(mutex_);
      if (capacity_refresh_cv_.wait_for(lock, kCapacityRefreshInterval, [this] {
            return !capacity_refresh_running_;
          })) {
        return;
      }
    }
  });
}

void DistributedVirtualMemoryCoordinator::stop_worker_capacity_refresh() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!capacity_refresh_running_) {
      return;
    }
    capacity_refresh_running_ = false;
  }
  capacity_refresh_cv_.notify_all();
  if (capacity_refresh_thread_.joinable()) {
    capacity_refresh_thread_.join();
  }
}

bool DistributedVirtualMemoryCoordinator::map_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  return broadcast_map_to_kv_tensors(model_id, dp_rank, offsets);
}

bool DistributedVirtualMemoryCoordinator::unmap_pages(
    const std::string& model_id,
    int32_t dp_rank,
    const std::vector<int64_t>& offsets) {
  return broadcast_unmap_from_kv_tensors(model_id, dp_rank, offsets);
}

bool DistributedVirtualMemoryCoordinator::reserve_weight_pages(
    const std::string& model_id,
    size_t num_pages) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(budget_initialized_);
    // Preserve existing model accounting if a duplicate initialization arrives.
    if (weight_page_counts_.find(model_id) != weight_page_counts_.end() ||
        !page_budget_.try_reserve(
            0, model_worker_count_locked(model_id), num_pages)) {
      return false;
    }
    weight_page_counts_[model_id] = num_pages;
    update_memory_usage_locked();
  }
  if (num_pages == 0 || broadcast_alloc_weight_pages(model_id, num_pages)) {
    return true;
  }
  // The broadcast returns false only after all workers acknowledge rollback.
  std::lock_guard<std::mutex> lock(mutex_);
  page_budget_.release(0, model_worker_count_locked(model_id), num_pages);
  weight_page_counts_.erase(model_id);
  update_memory_usage_locked();
  return false;
}

void DistributedVirtualMemoryCoordinator::set_weight_page_count(
    const std::string& model_id,
    size_t num_pages) {
  std::lock_guard<std::mutex> lock(mutex_);
  weight_page_counts_[model_id] = num_pages;
}

size_t DistributedVirtualMemoryCoordinator::get_weight_page_count(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = weight_page_counts_.find(model_id);
  return it == weight_page_counts_.end() ? 0 : it->second;
}

void DistributedVirtualMemoryCoordinator::release_weight_pages(
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

bool DistributedVirtualMemoryCoordinator::restore_weight_pages(
    const std::string& model_id) {
  const size_t pages = get_weight_page_count(model_id);
  return pages == 0 || broadcast_alloc_weight_pages(model_id, pages);
}

bool DistributedVirtualMemoryCoordinator::try_reserve_model_memory(
    const std::string& model_id,
    const KVCacheMappingPlan& plan) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(budget_initialized_);
  const bool reserved =
      page_budget_.try_reserve(model_memory_demand_locked(model_id, plan));
  update_memory_usage_locked();
  return reserved;
}

void DistributedVirtualMemoryCoordinator::release_model_memory_budget(
    const std::string& model_id,
    const KVCacheMappingPlan& plan) {
  std::lock_guard<std::mutex> lock(mutex_);
  page_budget_.release(model_memory_demand_locked(model_id, plan));
  update_memory_usage_locked();
}

std::vector<size_t>
DistributedVirtualMemoryCoordinator::model_memory_demand_locked(
    const std::string& model_id,
    const KVCacheMappingPlan& plan) const {
  std::vector<size_t> demand(page_budget_.worker_count(), 0);
  for (size_t rank = 0; rank < plan.dp_group_page_ids.size(); ++rank) {
    const auto& ids = plan.dp_group_page_ids[rank];
    if (ids.empty()) {
      continue;
    }
    const auto [first, last] =
        dp_worker_range_locked(model_id, static_cast<int32_t>(rank));
    CHECK_LT(first, last);
    CHECK_GT(plan.physical_pages_per_virtual_page, 0);
    CHECK_LE(ids.size(),
             std::numeric_limits<size_t>::max() /
                 plan.physical_pages_per_virtual_page);
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
    CHECK_LE(weight_pages, std::numeric_limits<size_t>::max() - demand[worker]);
    demand[worker] += weight_pages;
  }
  return demand;
}

void DistributedVirtualMemoryCoordinator::release_kv_mapping_budget(
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

std::vector<size_t>
DistributedVirtualMemoryCoordinator::get_worker_free_page_counts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.free_pages_snapshot();
}

size_t DistributedVirtualMemoryCoordinator::get_free_pages_for_model(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return page_budget_.min_free_pages(0, model_worker_count_locked(model_id));
}

void DistributedVirtualMemoryCoordinator::update_memory_usage_locked() const {
  VLOG(2) << "Shared worker budget: total_pages=" << page_budget_.total_pages()
          << ", min_free_pages="
          << page_budget_.min_free_pages(0, page_budget_.worker_count());
}

}  // namespace xllm

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

#include "core/distributed_runtime/kv_cache_capacity_planner.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <tuple>

#include "core/common/device_monitor.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"
#include "models/model_registry.h"

namespace xllm {

KVCacheCapacityPlanner::KVCacheCapacityPlanner(
    const ModelArgs& model_args,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t world_size,
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    bool is_multimodal,
    int64_t reserved_memory_bytes)
    : model_args_(model_args),
      options_(options),
      worker_clients_(worker_clients),
      dtype_(dtype),
      world_size_(world_size),
      is_multimodal_(is_multimodal),
      reserved_memory_bytes_(reserved_memory_bytes) {}

KVCacheEstimateContext KVCacheCapacityPlanner::observe_memory() const {
  KVCacheEstimateContext context;
  context.dtype = dtype_;
  context.world_size = world_size_;
  context.is_multimodal = is_multimodal_;
  context.reserved_memory_bytes = reserved_memory_bytes_;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args_.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (config.enable_vmm_kvcache() && phy_pool.is_initialized()) {
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
        config.phy_page_granularity_size();
  } else {
    CHECK(!worker_clients_.empty()) << "KV cache estimation requires workers";
    std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
    futures.reserve(worker_clients_.size());
    for (const auto& worker : worker_clients_) {
      futures.emplace_back(worker->estimate_kv_cache_capacity_async());
    }
    auto results = folly::collectAll(futures).get();
    context.worker_memory.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      CHECK(results[i].hasValue())
          << "Failed to estimate KV cache capacity for worker: " << i;
      const auto& [available_memory, total_memory] = results[i].value();
      context.worker_memory.emplace_back(
          KVCacheMemorySnapshot{available_memory, total_memory});
    }
    if (config.enable_vmm_kvcache()) {
      const int64_t encoder_bytes =
          is_multimodal_ ? options_.max_encoder_cache_size() * 1024 * 1024 : 0;
      int64_t initial_bytes = std::numeric_limits<int64_t>::max();
      for (const KVCacheMemorySnapshot& memory : context.worker_memory) {
        const VmmArenaCapacity arena = calculate_vmm_arena_capacity(
            memory.available_memory,
            memory.total_memory,
            config.max_memory_utilization(),
            config.max_memory_utilization_configured(),
            config.vmm_elastic(),
            config.max_cache_size(),
            reserved_memory_bytes_ + encoder_bytes);
        initial_bytes = std::min(initial_bytes, arena.initial_bytes);
      }
      // Persistent states must fit startup memory, rather than the larger
      // virtual arena that an elastic pool can grow into later.
      context.virtual_memory_cache_size = initial_bytes;
    }
  }
  return context;
}

void KVCacheCapacityPlanner::prepare_virtual_memory(
    KVCacheCapacity& capacity) const {
  auto& coordinator = DistributedVirtualMemoryCoordinator::get_instance();
  coordinator.prepare_kv_cache_pool(options_,
                                    world_size_,
                                    worker_clients_.size(),
                                    capacity.linear_cache_size_in_bytes(),
                                    is_multimodal_);
  const VmmKVCacheLayout layout(
      capacity, model_args_, world_size_, dtype_, options_.is_draft_engine());
  const size_t total_pages = coordinator.get_num_total_phy_pages();
  const size_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  capacity
      .n_blocks(
          static_cast<int64_t>(layout.block_capacity(total_pages, page_size)))
      .cache_size_in_bytes(static_cast<int64_t>(total_pages) * page_size +
                           capacity.linear_cache_size_in_bytes());
}

KVCacheCapacity KVCacheCapacityPlanner::plan(
    bool initialize_virtual_memory) const {
  const KVCacheEstimateContext context = observe_memory();
  KVCacheCapacity capacity =
      KVCacheEstimator(model_args_).estimate(options_, context);
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      initialize_virtual_memory) {
    prepare_virtual_memory(capacity);
  }
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
  for (const auto& device : options_.devices()) {
    DeviceMonitor::get_instance().set_total_kv_cache_memory(
        device.index(), capacity.cache_size_in_bytes());
    DeviceMonitor::get_instance().set_total_activation_memory(device.index());
  }
  return capacity;
}

}  // namespace xllm

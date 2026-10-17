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

#include "core/framework/block/kv_cache_manager_factory.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

#include "core/common/device_monitor.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/virtual_kv_cache_page_allocator_impl.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"
#include "models/model_registry.h"

namespace xllm {

KVCacheCapacity KVCacheManagerFactory::estimate_capacity(
    const ModelArgs& model_args,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t world_size,
    const std::vector<std::shared_ptr<WorkerClient>>& worker_clients,
    bool is_multimodal) {
  KVCacheEstimateContext context;
  context.dtype = dtype;
  context.world_size = world_size;
  context.is_multimodal = is_multimodal;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  if (config.enable_virtual_memory() && !is_multimodal) {
    const auto& phy_pool = PhysicalPagePool::get_instance();
    CHECK(phy_pool.is_initialized()) << "PhysicalPagePool not initialized";
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
        config.phy_page_granularity_size();
  } else {
    CHECK(!worker_clients.empty()) << "KV cache estimation requires workers";
    std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
    futures.reserve(worker_clients.size());
    for (const auto& worker : worker_clients) {
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
  }

  KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(options, context);
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
  for (const auto& device : options.devices()) {
    DeviceMonitor::get_instance().set_total_kv_cache_memory(
        device.index(), capacity.cache_size_in_bytes());
    DeviceMonitor::get_instance().set_total_activation_memory(device.index());
  }
  return capacity;
}

KVCacheManagerFactoryResult KVCacheManagerFactory::create(
    const KVCacheCapacity& kv_cache_capacity,
    const ModelArgs& model_args,
    int64_t world_size,
    torch::ScalarType dtype,
    BlockManagerPool::Options options,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
    int32_t dp_size,
    std::optional<HostCacheValidationOptions> host_validation_options) {
  CHECK_GT(world_size, 0) << "world_size must be greater than 0";
  CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";
#if defined(USE_NPU)
  CHECK(!options.enable_virtual_memory() ||
        !use_npu_nz_kv_cache_layout(model_args.model_type()))
      << "Virtual memory does not support the NPU NZ KV cache layout.";
#endif

  KVCacheShape shape(kv_cache_capacity, model_args, world_size);
  CHECK(shape.has_key_cache_shape())
      << "KV cache shape must contain a key cache shape";
  const std::vector<int64_t>& key_cache_shape = shape.key_cache_shape();
  CHECK(!key_cache_shape.empty()) << "KV cache shape cannot be empty";
  CHECK_GE(key_cache_shape.front(), 0)
      << "KV cache embedding block count cannot be negative";
  CHECK_LE(key_cache_shape.front(),
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
      << "KV cache embedding block count exceeds uint32_t range";
  options.num_embedding_blocks(static_cast<uint32_t>(key_cache_shape.front()));

  if (options.enable_linear_state()) {
    CHECK_LE(kv_cache_capacity.num_linear_state_blocks(),
             static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        << "Linear state slot count exceeds int32_t range";
    options.linear_state_num_slots(
        static_cast<int32_t>(kv_cache_capacity.num_linear_state_blocks()));
  }

  if (host_validation_options.has_value()) {
    HostCacheValidationOptions& validation = *host_validation_options;
    validation.device_block_count = kv_cache_capacity.n_blocks();
    validation.has_key_cache_shape = shape.has_key_cache_shape();
    validation.has_grouped_cache_layout = shape.has_grouped_cache_layout();
    validation.has_conv_cache_shape = shape.has_conv_cache_shape();
    validation.has_ssm_cache_shape = shape.has_ssm_cache_shape();
    const std::optional<std::string> validation_error =
        validate_host_cache_options(validation);
    if (validation_error.has_value()) {
      LOG(FATAL) << *validation_error;
    }
  }

  VirtualKVCachePageAllocator* page_allocator = nullptr;
  if (options.enable_virtual_memory()) {
    CHECK(shape.has_value_cache_shape());
    CHECK_EQ(kv_cache_capacity.scale_slot_size(), 0)
        << "Virtual memory does not support quantized KV cache";
    CHECK(!kv_cache_capacity.enable_mla_kv_cache_quant());
    CHECK(!shape.has_grouped_cache_layout());
    CHECK(!shape.has_conv_cache_shape() && !shape.has_ssm_cache_shape());
    CHECK(!kv_cache_capacity.enable_indexer_cache_quant() &&
          !shape.has_index_cache_scale_shape())
        << "Virtual memory does not support quantized indexer cache";
    CHECK(!model_args.index_kpool_compress())
        << "Virtual memory does not support KPool cache";
    const size_t element_size = torch::elementSize(dtype);
    const size_t key_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
        shape.key_cache_shape(), element_size);
    const size_t value_bytes = VirtualKVCachePageLayout::tensor_block_bytes(
        shape.value_cache_shape(), element_size);
    CHECK_EQ(shape.value_cache_shape().front(), key_cache_shape.front())
        << "Virtual memory requires matching K/V block counts";
    CHECK_GT(options.num_layers(), 0);
    const size_t num_layers = static_cast<size_t>(options.num_layers());
    std::vector<size_t> block_bytes{key_bytes, value_bytes};
    std::vector<size_t> tensor_counts{num_layers, num_layers};
    if (shape.has_index_cache_shape()) {
      CHECK_EQ(dtype, torch::kBFloat16)
          << "Virtual memory indexer cache requires BF16";
      CHECK_EQ(shape.index_cache_shape().front(), key_cache_shape.front())
          << "Virtual memory requires matching K/V/INDEX block counts";
      const std::vector<bool> indexer_layers =
          resolve_indexer_cache_enabled_layers(model_args,
                                               options.num_layers());
      CHECK(indexer_layers.empty() || indexer_layers.size() == num_layers);
      const size_t indexer_count =
          indexer_layers.empty()
              ? num_layers
              : static_cast<size_t>(std::count(
                    indexer_layers.begin(), indexer_layers.end(), true));
      if (indexer_count > 0) {
        block_bytes.emplace_back(VirtualKVCachePageLayout::tensor_block_bytes(
            shape.index_cache_shape(), element_size));
        tensor_counts.emplace_back(indexer_count);
      }
    }
    auto& allocator = DistributedVirtualMemoryCoordinator::get_instance()
                          .virtual_kv_cache_page_allocator();
    if (options.enable_disagg_pd()) {
      for (size_t bytes : block_bytes) {
        CHECK_LE(bytes, allocator.page_size())
            << "Virtual memory PD requires each tensor block to fit one page";
        CHECK_EQ(allocator.page_size() % bytes, 0)
            << "Virtual memory PD tensor blocks must divide the page size";
      }
    }
    const VirtualKVCachePageLayout layout(std::move(block_bytes),
                                          allocator.page_size());
    const size_t physical_pages = layout.physical_pages(tensor_counts);
    // Leave the tensor shape unchanged, but expose only complete page groups
    // to the scheduler. This prevents mapping past a short final tensor page.
    const size_t virtual_pages =
        static_cast<size_t>(kv_cache_capacity.n_blocks()) /
        layout.blocks_per_page();
    CHECK_GT(virtual_pages, 0)
        << "KV cache must fit one complete VM page group";
    CHECK(allocator.register_model(options.model_id(),
                                   options.num_layers(),
                                   physical_pages,
                                   virtual_pages));
    const size_t usable_pages =
        allocator.get_num_total_virt_pages(options.model_id());
    CHECK_GT(usable_pages, 0) << "Physical pool cannot fit one VM page group";
    const size_t usable_blocks = usable_pages * layout.blocks_per_page();
    LOG(INFO) << "Virtual KV cache: blocks=" << usable_blocks
              << ", blocks_per_page=" << layout.blocks_per_page()
              << ", physical_pages_per_group=" << physical_pages;
    options.virtual_block_bytes(layout.block_bytes())
        .num_blocks(static_cast<uint32_t>(usable_blocks));
    page_allocator = &allocator;
  }
  std::unique_ptr<KVCacheManager> manager;
  if (options.enable_host_offload()) {
    CHECK(transfer_coordinator != nullptr)
        << "KV cache transfer coordinator is required for host-offload KV "
           "cache manager";
    manager = std::make_unique<HierarchyBlockManagerPool>(
        options, std::move(transfer_coordinator), dp_size, page_allocator);
  } else {
    manager =
        std::make_unique<BlockManagerPool>(options, dp_size, page_allocator);
  }

  return KVCacheManagerFactoryResult{std::move(manager), std::move(shape)};
}

}  // namespace xllm

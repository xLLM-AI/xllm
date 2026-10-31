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

#include <limits>
#include <utility>

#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"

namespace xllm {

KVCacheManagerFactoryResult KVCacheManagerFactory::create(
    const KVCacheCapacity& kv_cache_capacity,
    const ModelArgs& model_args,
    int64_t world_size,
    torch::ScalarType dtype,
    BlockManagerPool::Options options,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator,
    int32_t dp_size,
    std::optional<HostCacheValidationOptions> host_validation_options,
    const VmmKVCacheLayout* shared_vmm_layout) {
  CHECK_GT(world_size, 0) << "world_size must be greater than 0";
  CHECK_GT(dp_size, 0) << "dp_size must be greater than 0";
  CHECK(!options.enable_vmm_kvcache() ||
        (!options.enable_host_offload() &&
         (!host_validation_options.has_value() ||
          host_validation_options->host_blocks_factor <= 1.0)))
      << "Virtual memory does not support host KV cache offload; set "
         "--host_blocks_factor=0.";
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
    validation.has_kpool_tail_shape = shape.has_kpool_tail_shape();
    validation.requires_indexer_row_folding =
        shape.has_index_cache_shape() &&
        shape.index_cache_shape().front() != key_cache_shape.front();
    const std::optional<std::string> validation_error =
        validate_host_cache_options(validation);
    if (validation_error.has_value()) {
      LOG(FATAL) << *validation_error;
    }
  }

  VmmKVPageAllocator* page_allocator = nullptr;
  if (options.enable_vmm_kvcache()) {
    std::optional<VmmKVCacheLayout> component_layout;
    if (shared_vmm_layout == nullptr) {
      component_layout.emplace(
          kv_cache_capacity, model_args, world_size, dtype);
    }
    const VmmKVCacheLayout& tensors =
        shared_vmm_layout == nullptr ? *component_layout : *shared_vmm_layout;
    auto& allocator = DistributedVirtualMemoryCoordinator::get_instance()
                          .vmm_kv_page_allocator();
    if (options.enable_disagg_pd()) {
      tensors.validate_pd_page_size(allocator.page_size());
    }
    const VirtualKVCachePageLayout layout =
        tensors.page_layout(allocator.page_size());
    const size_t physical_pages = tensors.physical_pages(allocator.page_size());
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

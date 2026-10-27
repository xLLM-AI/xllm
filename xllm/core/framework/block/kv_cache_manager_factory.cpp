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
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/speculative/mtp_utils.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/virtual_kv_cache_page_layout.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

// Shared capacity estimation for one model side (target or draft): resolves
// the memory budget, then the per-layer cache blocks. File-local (anonymous
// namespace) so the KVCacheCapacity layout and NPU build guards stay out of
// the public factory header. Reuses a worker-memory snapshot already polled
// by the caller (empty when virtual memory or multimodal mode supplies the
// budget differently).
KVCacheCapacity estimate_capacity(
    const ModelArgs& model_args,
    const runtime::Options& options,
    torch::ScalarType dtype,
    int64_t world_size,
    const std::vector<KVCacheMemorySnapshot>& worker_memory,
    bool is_multimodal,
    int64_t reserved_memory_bytes) {
  KVCacheEstimateContext context;
  context.dtype = dtype;
  context.world_size = world_size;
  context.is_multimodal = is_multimodal;
  context.reserved_memory_bytes = reserved_memory_bytes;
  context.linear_state_cache_block_limit =
      get_npu_linear_state_cache_block_limit(model_args.model_type());
  const KVCacheConfig& config = KVCacheConfig::get_instance();
  const bool vmm_replaces_memory =
      vmm_budget_replaces_worker_memory(is_multimodal);
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (vmm_replaces_memory && phy_pool.is_initialized()) {
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
        config.phy_page_granularity_size();
  } else {
    context.worker_memory = worker_memory;
    if (vmm_replaces_memory) {
      // The pool is not up yet: bound the estimate by every worker's VMM
      // arena startup capacity; the pool is sized from the final estimate.
      const int64_t encoder_bytes =
          is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024 : 0;
      int64_t initial_bytes = std::numeric_limits<int64_t>::max();
      for (const KVCacheMemorySnapshot& memory : context.worker_memory) {
        const VmmArenaCapacity arena =
            calculate_vmm_arena_capacity(memory.available_memory,
                                         memory.total_memory,
                                         config.max_memory_utilization(),
                                         config.vmm_elastic(),
                                         config.max_cache_size(),
                                         reserved_memory_bytes + encoder_bytes);
        initial_bytes = std::min(initial_bytes, arena.initial_bytes);
      }
      // Persistent states must fit startup memory, rather than the larger
      // virtual arena that an elastic pool can grow into later.
      context.virtual_memory_cache_size = initial_bytes;
    }
  }
  return KVCacheEstimator(model_args).estimate(options, context);
}

// True when the draft gets no separate capacity of its own: the target
// capacity passes through unchanged, so estimating a draft KV budget would be
// wasted work. Covers the embedded Eagle3 draft (the draft model stays
// inside the target engine with a separate shape) and per-type compressed
// block pools (the draft shares the target's pool geometry), detected from
// the capacity data itself so any model with such pools inherits it.
bool target_draft_capacity_passes_through(
    const ModelArgs& target_args,
    std::string_view speculative_algorithm,
    const KVCacheCapacity& target_cap) {
  return uses_embedded_eagle3_draft(speculative_algorithm, target_args) ||
         target_cap.c4_count() > 0 || target_cap.c128_count() > 0;
}

}  // namespace

KVCacheCapacity KVCacheManagerFactory::estimate_speculative_capacity(
    const KVCacheModelEstimate& target,
    const std::optional<KVCacheModelEstimate>& draft,
    int64_t world_size,
    const std::vector<KVCacheMemorySnapshot>& worker_memory,
    bool is_multimodal,
    int64_t reserved_memory_bytes) {
  KVCacheCapacity result = estimate_capacity(target.args,
                                             target.options,
                                             target.dtype,
                                             world_size,
                                             worker_memory,
                                             is_multimodal,
                                             reserved_memory_bytes);
  // A draft side that gets no separate capacity of its own (embedded Eagle3
  // draft, per-type compressed pools) passes the target capacity through
  // unchanged; a draft estimator run would be discarded. Otherwise the draft
  // model is always a plain text LLM even under a VLM target, so its
  // estimation stays on the text path and the encoder-cache reservation is
  // only deducted once (by the target).
  const bool estimate_draft =
      draft.has_value() &&
      !target_draft_capacity_passes_through(
          target.args, target.options.speculative_algorithm(), result);
  if (estimate_draft) {
    const KVCacheCapacity draft_cap = estimate_capacity(draft->args,
                                                        draft->options,
                                                        draft->dtype,
                                                        world_size,
                                                        worker_memory,
                                                        /*is_multimodal=*/false,
                                                        reserved_memory_bytes);
    result = combine_target_and_draft_kv_cache_capacity(
        result, target.args, draft->args, draft_cap);
  }
  return result;
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
    const VmmKVCacheLayout tensors(
        kv_cache_capacity, model_args, world_size, dtype);
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
  std::unique_ptr<BlockManagerPool> manager;
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

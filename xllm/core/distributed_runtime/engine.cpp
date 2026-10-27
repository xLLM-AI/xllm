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

#include "core/distributed_runtime/engine.h"

#include "core/common/device_monitor.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/distributed_runtime/worker_fanout.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "framework/block/kv_cache_manager_factory.h"
#include "framework/model_loader/model_loader.h"
#include "models/model_registry.h"
#include "runtime/spec_runtime_helpers.h"
#include "util/utils.h"

namespace xllm {
namespace {

// Speculative decoding (including the draft-free Suffix algorithm) sizes
// workers for num_speculative_tokens + 1 decode tokens per step.
runtime::Options make_speculative_worker_options(
    const runtime::Options& options) {
  auto worker_options = options;
  if (SpeculativeConfig::uses_speculative_pipeline(options)) {
    worker_options.num_decoding_tokens(options.num_speculative_tokens() + 1)
        .enable_speculative_decode(/*value=*/true);
  }
  return worker_options;
}

// Sizes the shared VMM page pool from the final capacity (persistent state
// included) and registers the model's parallel strategy.
KVCacheCapacity prepare_virtual_memory_pool(KVCacheCapacity capacity,
                                            const ModelArgs& model_args,
                                            const runtime::Options& options,
                                            torch::ScalarType dtype,
                                            int64_t world_size,
                                            int64_t num_workers,
                                            bool is_multimodal) {
  auto& coordinator = DistributedVirtualMemoryCoordinator::get_instance();
  coordinator.prepare_kv_cache_pool(options,
                                    world_size,
                                    static_cast<size_t>(num_workers),
                                    capacity.linear_cache_size_in_bytes(),
                                    is_multimodal);
  const VmmKVCacheLayout layout(
      capacity, model_args, world_size, dtype, options.is_draft_model());
  const size_t total_pages = coordinator.get_num_total_phy_pages();
  const size_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  capacity
      .n_blocks(
          static_cast<int64_t>(layout.block_capacity(total_pages, page_size)))
      .cache_size_in_bytes(static_cast<int64_t>(total_pages) * page_size +
                           capacity.linear_cache_size_in_bytes());
  return capacity;
}

// Reports the final (post-combine) capacity so the KV memory gauges never
// carry an intermediate per-model estimate.
void report_capacity_metrics(const KVCacheCapacity& capacity,
                             const runtime::Options& options) {
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
  for (const auto& device : options.devices()) {
    DeviceMonitor::get_instance().set_total_kv_cache_memory(
        device.index(), capacity.cache_size_in_bytes());
    DeviceMonitor::get_instance().set_total_activation_memory(device.index());
  }
}

}  // namespace

bool Engine::dispatch_model_load(const ModelLoader& model_loader,
                                 const ModelArgs& model_args,
                                 MasterStatus master_status,
                                 const ModelInitCallback& prepare_model) {
  if (prepare_model &&
      !prepare_model(model_loader,
                     model_args.n_layers(),
                     /*dp_size=*/static_cast<int32_t>(dp_size()),
                     static_cast<int32_t>(dp_local_tp_size_),
                     master_status)) {
    return false;
  }
  return load_model(
      worker_clients_, model_loader.model_weights_path(), master_status);
}

bool Engine::init_draft_model(const ModelInitCallback& prepare_model) {
  const std::string& draft_model_path = options_.draft_model_path().value();
  auto model_loader = ModelLoader::create(draft_model_path);
  ModelArgs draft_args = model_loader->model_args();
  ModelRegistry::configure_mtp_args(
      draft_args,
      options_.speculative_algorithm(),
      /*is_draft_model=*/true,
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl()));
  LOG(INFO) << "Initializing draft model from: " << draft_model_path;

  // Draft tokens are detokenized by the target, so the draft model needs no
  // tokenizer and no draft/target vocab-compatibility check.
  if (args_.max_position_embeddings() != draft_args.max_position_embeddings()) {
    LOG(WARNING) << "draft and target models have different max context "
                    "lengths, draft max_position_embeddings: "
                 << draft_args.max_position_embeddings()
                 << ", target max_position_embeddings: "
                 << args_.max_position_embeddings()
                 << ", using the minimum between them";
    args_.max_position_embeddings() = std::min(
        args_.max_position_embeddings(), draft_args.max_position_embeddings());
  }

  if (!dispatch_model_load(
          *model_loader, draft_args, MasterStatus::WAKEUP, prepare_model)) {
    return false;
  }
  draft_args_ = std::move(draft_args);
  return true;
}

runtime::Options Engine::draft_estimate_options() const {
  runtime::Options draft_options = options_;
  spec_detail::mark_spec_draft_options(draft_options);
  draft_options.model_path(options_.draft_model_path().value())
      .backend("llm")
      .num_decoding_tokens(1)
      .enable_speculative_decode(/*value=*/false)
      .enable_graph(/*value=*/false);
  return draft_options;
}

std::optional<KVCacheCapacity> Engine::init_draft_and_estimate_capacity(
    const ModelInitCallback& prepare_model) {
  std::optional<KVCacheModelEstimate> draft;
  if (SpeculativeConfig::loads_draft_model(options_)) {
    // Virtual memory hosts one model_id, so an external draft is rejected
    // before any draft checkpoint load is fanned out to the workers.
    if (KVCacheConfig::get_instance().enable_vmm_kvcache() ||
        LoadConfig::get_instance().enable_vmm_weights()) {
      LOG(ERROR) << "Virtual memory does not support an external speculative "
                    "draft engine because target and draft would register the "
                    "same model_id.";
      return std::nullopt;
    }
    if (!init_draft_model(prepare_model)) {
      return std::nullopt;
    }
    // Keep the per-model dtype parsing for mixed-dtype drafts.
    const runtime::Options draft_options = draft_estimate_options();
    draft.emplace(
        draft_args_,
        draft_options,
        util::parse_dtype(draft_args_.dtype(), options_.devices()[0]));
  }
  const bool vmm_replaces_memory =
      vmm_budget_replaces_worker_memory(is_multimodal());
  // Once the VMM pool is initialized its page budget replaces the worker
  // snapshots entirely, so the poll (an RPC per worker) is skipped.
  const std::vector<KVCacheMemorySnapshot> worker_memory =
      vmm_replaces_memory && PhysicalPagePool::get_instance().is_initialized()
          ? std::vector<KVCacheMemorySnapshot>{}
          : poll_worker_memory(worker_clients_);
  KVCacheCapacity result = KVCacheManagerFactory::estimate_speculative_capacity(
      {args_, options_, dtype_},
      draft,
      dp_local_tp_size_,
      worker_memory,
      is_multimodal(),
      reserved_memory_bytes_);
  if (vmm_replaces_memory) {
    result = prepare_virtual_memory_pool(
        std::move(result),
        args_,
        options_,
        dtype_,
        dp_local_tp_size_,
        static_cast<int64_t>(worker_clients_.size()),
        is_multimodal());
  }
  report_capacity_metrics(result, options_);
  return result;
}

void Engine::process_group_test() {
#if !defined(USE_NPU)
  // In multi-node serving mode, only driver engine creates worker_clients_.
  fanout_process_group_test(worker_clients_);
#endif
}

void Engine::setup_workers() {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ = std::make_shared<DistributedWorkerManager>(
        make_speculative_worker_options(options_));
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

}  // namespace xllm

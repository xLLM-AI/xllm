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

#include <chrono>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_vmm_coordinator.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/draft_config.h"
#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "core/util/env_var.h"
#include "core/util/pretty_print.h"
#include "core/util/tensor_options.h"
#include "framework/model_loader/model_loader.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

std::vector<KVCacheMemorySnapshot> poll_worker_memory(
    const std::vector<std::shared_ptr<WorkerClient>>& workers) {
  CHECK(!workers.empty()) << "KV cache estimation requires workers";
  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(workers.size());
  for (auto& worker : workers) {
    futures.emplace_back(worker->estimate_kv_cache_capacity_async());
  }
  const auto results = folly::collectAll(futures).get();
  std::vector<KVCacheMemorySnapshot> worker_memory;
  worker_memory.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    CHECK(results[i].hasValue())
        << "Failed to estimate KV cache capacity for worker: " << i;
    const auto& [available_memory, total_memory] = results[i].value();
    worker_memory.emplace_back(
        KVCacheMemorySnapshot{available_memory, total_memory});
  }
  return worker_memory;
}

void fanout_process_group_test(
    const std::vector<std::shared_ptr<WorkerClient>>& workers) {
#if defined(USE_NPU)
  (void)workers;
#else
  if (workers.size() <= 1) {
    return;
  }
  std::vector<folly::SemiFuture<folly::Unit>> futures;
  futures.reserve(workers.size());
  for (auto& worker : workers) {
    futures.emplace_back(worker->process_group_test_async());
  }
  const auto timeout_seconds = util::get_process_group_test_timeout_seconds();
  folly::collectAll(futures)
      .within(std::chrono::seconds(timeout_seconds))
      .get();
#endif
}

runtime::Options make_speculative_worker_options(
    const runtime::Options& options) {
  if (!SpeculativeConfig::uses_speculative_pipeline(options)) {
    return options;
  }
  auto worker_options = options;
  worker_options.enable_speculative_decode(/*value=*/true);
  return worker_options;
}

KVCacheCapacity prepare_virtual_memory_pool(KVCacheCapacity capacity,
                                            const ModelArgs& model_args,
                                            const runtime::Options& options,
                                            torch::ScalarType dtype,
                                            int64_t world_size,
                                            int64_t num_workers,
                                            bool is_multimodal) {
  auto& coordinator = DistributedVmmCoordinator::get_instance();
  coordinator.prepare_kv_cache_pool(options,
                                    world_size,
                                    static_cast<size_t>(num_workers),
                                    capacity.linear_cache_size_in_bytes(),
                                    is_multimodal);
  const VmmKVCacheLayout layout(capacity, model_args, world_size, dtype);
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

void report_capacity_metrics(const KVCacheCapacity& capacity) {
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            capacity.cache_size_in_bytes() / 1024);
}

runtime::Options draft_estimate_options(const runtime::Options& options) {
  runtime::Options draft_options = options;
  spec_detail::mark_spec_draft_options(draft_options);
  draft_options.model_path(options.draft_model_path().value())
      .enable_speculative_decode(/*value=*/false)
      .enable_graph(/*value=*/false);
  return draft_options;
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
  return load_model_to_workers(model_loader.model_weights_path(),
                               master_status);
}

bool Engine::load_model_to_workers(const std::string& model_path,
                                   MasterStatus master_status) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(
        worker->init_model_async(model_path,
                                 ExecutionConfig::get_instance().random_seed(),
                                 master_status));
  }
  const auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
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

std::optional<KVCacheCapacity> Engine::init_draft_and_estimate_capacity(
    const ModelInitCallback& prepare_model) {
  std::optional<KVCacheModelEstimate> draft;
  if (SpeculativeConfig::loads_draft_model(options_)) {
    if (KVCacheConfig::get_instance().enable_vmm_kvcache() ||
        LoadConfig::get_instance().enable_vmm_weights()) {
      LOG(ERROR) << "Virtual memory does not support an external speculative "
                    "draft model because target and draft would register the "
                    "same model_id.";
      return std::nullopt;
    }
    if (!init_draft_model(prepare_model)) {
      return std::nullopt;
    }
    const runtime::Options draft_options = draft_estimate_options(options_);
    draft.emplace(
        draft_args_,
        draft_options,
        util::parse_dtype(draft_args_.dtype(), options_.devices()[0]));
  }
  const bool vmm_replaces_memory =
      vmm_budget_replaces_worker_memory(is_multimodal());
  const std::vector<KVCacheMemorySnapshot> worker_memory =
      vmm_pool_serves_budget(is_multimodal())
          ? std::vector<KVCacheMemorySnapshot>{}
          : poll_worker_memory(worker_clients_);
  KVCacheCapacity result =
      estimate_target_and_draft_capacity({args_, options_, dtype_},
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
  report_capacity_metrics(result);
  return result;
}

void Engine::process_group_test() {
  fanout_process_group_test(worker_clients_);
}

void Engine::log_capacity_and_validate_chunked_prefill(
    const KVCacheCapacity& kv_cache_cap,
    bool enable_state_cache) const {
  LOG(INFO) << "kv cache capacity: "
            << readable_size(kv_cache_cap.cache_size_in_bytes())
            << ", blocks: " << kv_cache_cap.n_blocks()
            << ", slot_size: " << kv_cache_cap.slot_size()
            << ", index_slot_size: " << kv_cache_cap.index_slot_size()
            << ", indexer_layers: " << kv_cache_cap.num_indexer_layers()
            << ", scale_slot_size: " << kv_cache_cap.scale_slot_size()
            << ", linear_slot_size: " << kv_cache_cap.linear_slot_size()
            << ", linear_blocks: " << kv_cache_cap.num_linear_state_blocks()
            << ", state_slots: "
            << (enable_state_cache ? kv_cache_cap.num_linear_state_blocks() : 0)
            << ", gdn_state_bytes: "
            << readable_size(kv_cache_cap.num_linear_attention_layers() *
                             kv_cache_cap.linear_slot_size())
            << ", kpool_state_bytes: "
            << readable_size(kv_cache_cap.num_indexer_layers() *
                             kv_cache_cap.kpool_tail_slot_size())
            << ", max_linear_state_cache_slots: "
            << options_.max_linear_state_cache_slots()
            << ", reserved_state_bytes: "
            << readable_size(kv_cache_cap.linear_cache_size_in_bytes())
            << ", n_layers: " << kv_cache_cap.n_layers()
            << ", kv_cache_dtype: " << options_.kv_cache_dtype();

  // DECODE disables the LINEAR prefix cache by role, and pd_launch.sh
  // legitimately sets --enable_chunked_prefill=false there.
  const bool is_decode = options_.instance_role() == InstanceRole::DECODE;
  if (options_.enable_prefix_cache() && enable_state_cache && !is_decode) {
    const auto& scheduler_config = ::xllm::SchedulerConfig::get_instance();
    const int32_t block_size = static_cast<int32_t>(kv_cache_cap.block_size());
    CHECK(scheduler_config.enable_chunked_prefill())
        << "GDN/KPool state prefix cache requires block-aligned chunked "
           "prefill to save matching linear states. Please set "
           "--enable_chunked_prefill=true in your config.";
    CHECK(scheduler_config.max_tokens_per_chunk_for_prefill() % block_size == 0)
        << "state-cache prefix cache saves checkpoints at "
           "chunk-end boundaries, so max_tokens_per_chunk_for_prefill ("
        << scheduler_config.max_tokens_per_chunk_for_prefill()
        << ") must be a multiple of block_size (" << block_size << ").";
  }
}

bool Engine::allocate_kv_cache_on_workers(const KVCacheShape& kv_cache_shape,
                                          bool use_transfer_path) {
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(
        use_transfer_path
            ? worker->allocate_kv_cache_with_transfer_async(kv_cache_shape)
            : worker->allocate_kv_cache_async(kv_cache_shape));
  }
  const auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  block_manager_pool_->reserve_virtual_memory_padding_blocks();
  return true;
}

void Engine::setup_workers() {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ = std::make_shared<DistributedWorkerManager>(
        make_speculative_worker_options(options_));
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

}  // namespace xllm

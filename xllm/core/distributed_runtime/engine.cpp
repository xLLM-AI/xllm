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

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/distributed_runtime/distributed_vmm_coordinator.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/allocator/vmm_arena_capacity.h"
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

KVCacheCapacity estimate_side_capacity(
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
  const auto& phy_pool = PhysicalPagePool::get_instance();
  if (config.enable_vmm_kvcache() && phy_pool.is_initialized()) {
    context.virtual_memory_cache_size =
        static_cast<int64_t>(phy_pool.num_total()) *
        config.phy_page_granularity_size();
  } else {
    context.worker_memory = worker_memory;
    if (config.enable_vmm_kvcache()) {
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
      // Persistent states must fit the initial physical pool.
      context.virtual_memory_cache_size = initial_bytes;
    }
  }
  return KVCacheEstimator(model_args).estimate(options, context);
}

}  // namespace

void Engine::reconcile_vocab_size(int64_t tokenizer_vocab_size) {
  const int64_t model_vocab_size = args_.vocab_size();
  if (tokenizer_vocab_size == model_vocab_size) {
    return;
  }
  if (model_vocab_size <= 0) {
    LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab size: "
                 << tokenizer_vocab_size;
    args_.vocab_size(tokenizer_vocab_size);
  } else if (tokenizer_vocab_size > model_vocab_size) {
    LOG(WARNING) << "Unsafe vocab mismatch: tokenizer: " << tokenizer_vocab_size
                 << ", model: " << model_vocab_size;
  } else {
    LOG(INFO) << "Tokenizer/model vocab differ: tokenizer="
              << tokenizer_vocab_size << ", model=" << model_vocab_size;
  }
}

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
  futures.reserve(worker_clients_num());
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
      /*is_draft_engine=*/true,
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

std::optional<Engine::KVCachePlan> Engine::init_draft_and_estimate_capacity(
    const ModelInitCallback& prepare_model) {
  const auto& config = KVCacheConfig::get_instance();
  const bool loads_draft = SpeculativeConfig::loads_draft_model(options_);
  if (loads_draft) {
    if (LoadConfig::get_instance().enable_vmm_weights()) {
      LOG(ERROR) << "Virtual weight memory does not support an external "
                    "speculative draft engine.";
      return std::nullopt;
    }
    if (config.enable_vmm_kvcache() &&
        (!SpeculativeConfig::is_mtp_algorithm(
             options_.speculative_algorithm()) ||
         options_.enable_disagg_pd() ||
         options_.instance_role() != InstanceRole::DEFAULT)) {
      LOG(ERROR) << "Virtual KV cache supports co-located MTP target and draft "
                    "engines only; PD is unsupported.";
      return std::nullopt;
    }
    if (!init_draft_model(prepare_model)) {
      return std::nullopt;
    }
  }

  const std::vector<KVCacheMemorySnapshot> worker_memory =
      config.enable_vmm_kvcache() &&
              PhysicalPagePool::get_instance().is_initialized()
          ? std::vector<KVCacheMemorySnapshot>{}
          : poll_worker_memory(worker_clients_);
  KVCacheCapacity target = estimate_side_capacity(args_,
                                                  options_,
                                                  dtype_,
                                                  dp_local_tp_size_,
                                                  worker_memory,
                                                  is_multimodal(),
                                                  reserved_memory_bytes_);
  std::optional<VmmKVCacheLayout> shared_vmm_layout;
  int64_t reserved_state_bytes = target.linear_cache_size_in_bytes();
  if (loads_draft && (config.enable_vmm_kvcache() ||
                      (target.c4_count() == 0 && target.c128_count() == 0))) {
    runtime::Options draft_options = options_;
    spec_detail::mark_spec_draft_options(draft_options);
    draft_options.model_path(options_.draft_model_path().value())
        .enable_speculative_decode(/*value=*/false)
        .enable_graph(/*value=*/false);
    const torch::ScalarType draft_dtype =
        util::parse_dtype(draft_args_.dtype(), options_.devices()[0]);
    const int64_t draft_world_size =
        options_.enable_mtp_draft_body_tp1() ? 1 : dp_local_tp_size_;
    const KVCacheCapacity draft =
        estimate_side_capacity(draft_args_,
                               draft_options,
                               draft_dtype,
                               draft_world_size,
                               worker_memory,
                               /*is_multimodal=*/false,
                               /*reserved_memory_bytes=*/0);
    reserved_state_bytes += draft.linear_cache_size_in_bytes();
    if (config.enable_vmm_kvcache()) {
      shared_vmm_layout.emplace(target, args_, dp_local_tp_size_, dtype_);
      shared_vmm_layout->append(VmmKVCacheLayout(draft,
                                                 draft_args_,
                                                 draft_world_size,
                                                 draft_dtype,
                                                 /*is_draft_engine=*/true));
    } else if (target.shared_slot_bytes() > 0) {
      target = plan_shared_static_kv_cache(
                   std::move(target), draft, args_, dp_local_tp_size_, dtype_)
                   .target;
    } else {
      target = combine_target_and_draft_kv_cache_capacity(
          target, args_, draft_args_, draft);
    }
  }
  if (config.enable_vmm_kvcache()) {
    auto& coordinator = DistributedVmmCoordinator::get_instance();
    coordinator.prepare_kv_cache_pool(
        options_,
        dp_local_tp_size_,
        static_cast<int64_t>(worker_clients_num()),
        reserved_state_bytes,
        is_multimodal());
    if (!shared_vmm_layout) {
      shared_vmm_layout.emplace(target, args_, dp_local_tp_size_, dtype_);
    }
    const size_t total_pages = coordinator.get_num_total_phy_pages();
    const size_t page_size = config.phy_page_granularity_size();
    target
        .n_blocks(static_cast<int64_t>(
            shared_vmm_layout->block_capacity(total_pages, page_size)))
        .cache_size_in_bytes(static_cast<int64_t>(total_pages) * page_size +
                             reserved_state_bytes);
  }
  GAUGE_SET(total_kv_cache_size_in_kilobytes,
            target.cache_size_in_bytes() / 1024);
  return KVCachePlan{std::move(target), std::move(shared_vmm_layout)};
}

void Engine::process_group_test() {
#if !defined(USE_NPU)
  if (worker_clients_.size() <= 1) {
    return;
  }
  std::vector<folly::SemiFuture<folly::Unit>> futures;
  futures.reserve(worker_clients_num());
  for (auto& worker : worker_clients_) {
    futures.emplace_back(worker->process_group_test_async());
  }
  const auto timeout_seconds = util::get_process_group_test_timeout_seconds();
  folly::collectAll(futures)
      .within(std::chrono::seconds(timeout_seconds))
      .get();
#endif
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
  futures.reserve(worker_clients_num());
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
    runtime::Options worker_options = options_;
    if (SpeculativeConfig::uses_speculative_pipeline(options_)) {
      worker_options.enable_speculative_decode(true);
    }
    distributed_worker_manager_ =
        std::make_shared<DistributedWorkerManager>(worker_options);
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

bool Engine::start_profile() {
  std::lock_guard<std::mutex> lock(profile_mutex_);
  LOG(INFO) << "Starting profiler on " << worker_clients_num() << " worker(s).";
  if (!profile_workers(/*is_start=*/true)) {
    // A partially successful broadcast must not leave other workers tracing.
    if (!profile_workers(/*is_start=*/false)) {
      LOG(ERROR) << "Failed to clean up profiling on some workers. "
                    "Retry /stop_profile.";
    }
    return false;
  }
  return true;
}

bool Engine::stop_profile() {
  std::lock_guard<std::mutex> lock(profile_mutex_);
  LOG(INFO) << "Stopping profiler on " << worker_clients_num() << " worker(s).";
  return profile_workers(/*is_start=*/false);
}

bool Engine::profile_workers(bool is_start) {
  if (worker_clients_.empty()) {
    LOG(ERROR) << "No worker clients available for profiling.";
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num());
  for (auto& worker : worker_clients_) {
    futures.emplace_back(is_start ? worker->start_profile_async()
                                  : worker->stop_profile_async());
  }

  auto results = folly::collectAll(futures).get();
  bool success = true;
  for (const auto& result : results) {
    if (result.hasException()) {
      LOG(ERROR) << "Profiling worker RPC failed: "
                 << result.exception().what();
      success = false;
      continue;
    }
    if (!result.value()) {
      LOG(ERROR) << "Profiling failed on a worker (start=" << is_start << ").";
      success = false;
    }
  }

  return success;
}

}  // namespace xllm

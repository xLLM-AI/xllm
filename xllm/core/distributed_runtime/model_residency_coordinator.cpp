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

#include "core/distributed_runtime/model_residency_coordinator.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_memory_coordinator.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/kv_cache/storage/paged_kv_cache_page_allocator.h"
#include "core/runtime/worker_memory_resources.h"

namespace xllm {

namespace {

constexpr size_t kWeightPageSafetyMargin = 20;

int64_t get_effective_weight_size(const ModelLoader& model_loader,
                                  int64_t num_layers) {
  constexpr int64_t kInvalidWeightSize = -1;
  const int64_t all_size = model_loader.get_total_weight_size();
  if (all_size <= 0) {
    LOG(ERROR)
        << "Invalid total model weight size: " << all_size
        << ". Ensure model .index.json exists and has metadata.total_size";
    return kInvalidWeightSize;
  }

  if (!LoadConfig::get_instance().enable_rolling_load()) {
    return all_size;
  }

  const int64_t non_decoder_size = model_loader.get_non_decoder_weight_size();
  if (non_decoder_size <= 0) {
    LOG(ERROR) << "Invalid non-decoder weight size: " << non_decoder_size;
    return kInvalidWeightSize;
  }
  if (non_decoder_size > all_size) {
    LOG(ERROR) << "non_decoder_weight_size (" << non_decoder_size
               << ") exceeds total_weight_size (" << all_size << ")";
    return kInvalidWeightSize;
  }
  if (num_layers <= 0) {
    LOG(ERROR) << "Invalid layer count: " << num_layers;
    return kInvalidWeightSize;
  }

  const int64_t all_decoder_size = all_size - non_decoder_size;
  const int64_t max_layer_size =
      model_loader.get_max_decoder_layer_weight_size();
  if (max_layer_size <= 0) {
    LOG(ERROR) << "Failed to get max decoder layer size for rolling load.";
    return kInvalidWeightSize;
  }
  const int64_t rolling_buffer_size =
      LoadConfig::get_instance().rolling_load_num_cached_layers() *
      max_layer_size;
  const int64_t total_weight_size = non_decoder_size + rolling_buffer_size;

  LOG(INFO) << "Rolling-load weight budget: total=" << all_size
            << ", non_decoder=" << non_decoder_size
            << ", all_decoder=" << all_decoder_size
            << ", max_layer=" << max_layer_size
            << ", rolling_buffer=" << rolling_buffer_size << " ("
            << LoadConfig::get_instance().rolling_load_num_cached_layers()
            << " slots x " << max_layer_size << " bytes/max-layer)"
            << ", effective=" << total_weight_size;
  return total_weight_size;
}

}  // namespace

ModelResidencyCoordinator::ModelResidencyCoordinator(
    Options options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(std::move(options)),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {}

bool ModelResidencyCoordinator::initialize_model(
    const ModelLoader& model_loader,
    int64_t num_layers,
    int32_t dp_size,
    int32_t tp_size,
    MasterStatus master_status) {
  if (!options_.enabled) {
    return true;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to initialize model residency.";
    return false;
  }

  CHECK_GT(num_layers, 0);
  CHECK_GT(dp_size, 0);
  CHECK_GT(tp_size, 0);
  auto& coordinator = DistributedMemoryCoordinator::get_instance();
  if (!coordinator.is_initialized()) {
    auto& phy_pool = PhysicalPagePool::get_instance();
    CHECK(phy_pool.is_initialized())
        << "PhysicalPagePool must precede the shared worker page budget";
    const int32_t worker_count = static_cast<int32_t>(
        distributed_worker_manager_->get_worker_clients().size());
    coordinator.init_page_budget(
        phy_pool.num_total(),
        dp_size,
        worker_count,
        KVCacheConfig::get_instance().phy_page_granularity_size(),
        /*enable_prealloc=*/true);
  }

  const std::string& model_id = options_.model_id;
  coordinator.set_model_parallel_strategy(model_id, dp_size, tp_size);
  coordinator.kv_cache_page_allocator().register_model(model_id, num_layers);

  const int64_t total_weight_size =
      get_effective_weight_size(model_loader, num_layers);
  if (total_weight_size < 0) {
    return false;
  }
  const int64_t weight_size_per_tp =
      (total_weight_size + tp_size - 1) / tp_size;
  const size_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  const size_t num_pages = (weight_size_per_tp + page_size - 1) / page_size +
                           kWeightPageSafetyMargin;

  LOG(INFO) << "Model weight page budget: total_weight_size="
            << total_weight_size << ", tp_size=" << tp_size
            << ", weight_size_per_tp=" << weight_size_per_tp
            << ", num_pages=" << num_pages
            << ", master_status=" << master_status;

  if (master_status == MasterStatus::WAKEUP) {
    if (!coordinator.reserve_weight_pages(model_id, num_pages)) {
      LOG(ERROR) << "Failed to allocate weight pages";
      return false;
    }
    LOG(INFO)
        << "master_status=0 (MasterStatus::WAKEUP): Allocated weight pages, "
           "will load to device";
  } else if (master_status == MasterStatus::LIGHT_SLEEP ||
             master_status == MasterStatus::DEEP_SLEEP) {
    coordinator.set_weight_page_count(model_id, num_pages);
    LOG(INFO) << "master_status=" << master_status
              << " (SLEEP): Recorded weight pages, num_pages=" << num_pages;
  }

  return true;
}

bool ModelResidencyCoordinator::finish_initialization(
    MasterStatus master_status) {
  if (!options_.enabled || master_status == MasterStatus::WAKEUP) {
    return true;
  }

  // KV cache allocation must finish before releasing initial resources.
  if (!suspend_memory(/*skip_weight_release=*/true)) {
    LOG(ERROR) << "Failed to sleep model " << options_.model_id
               << " after init";
    return false;
  }
  LOG(INFO) << "Model " << options_.model_id
            << " put to sleep after init (master_status=" << master_status
            << ")";
  return true;
}

void ModelResidencyCoordinator::get_virtual_memory_info(
    std::vector<size_t>& worker_free_phy_pages,
    std::unordered_map<std::string, std::vector<WeightSegment>>&
        model_weight_segments) const {
  if (!options_.enabled) {
    return;
  }

  auto& coordinator = DistributedMemoryCoordinator::get_instance();
  if (coordinator.is_initialized()) {
    worker_free_phy_pages = coordinator.get_worker_free_page_counts();
  }
  model_weight_segments =
      WorkerMemoryResources::get_instance().get_all_model_weight_segments();
}

bool ModelResidencyCoordinator::sleep(MasterStatus master_status) {
  if (!options_.enabled) {
    LOG(WARNING) << "sleep requires --enable_virtual_memory=true";
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to sleep.";
    return false;
  }

  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  LOG(INFO) << "Starting to sleep model " << options_.model_id
            << ". Worker clients count: " << worker_clients.size();

  // Release weight and KV cache pages before changing worker model state.
  if (!suspend_memory(/*skip_weight_release=*/false)) {
    LOG(ERROR) << "Memory suspension failed, aborting sleep flow";
    return false;
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());
  for (const auto& worker : worker_clients) {
    futures.emplace_back(worker->sleep_async(master_status));
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Sleep failed.";
      return false;
    }
  }
  return true;
}

bool ModelResidencyCoordinator::wakeup(const WakeupOptions& options) {
  if (!options_.enabled) {
    LOG(WARNING) << "wakeup requires --enable_virtual_memory=true";
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to wakeup.";
    return false;
  }

  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  LOG(INFO) << "Starting to wakeup model " << options_.model_id
            << ". Worker clients count: " << worker_clients.size();

  // Restore weight and KV cache pages before workers load or transfer weights.
  auto& coordinator = DistributedMemoryCoordinator::get_instance();
  auto& pages = coordinator.kv_cache_page_allocator();
  const auto plan = pages.begin_resume(options_.model_id);
  if (!plan.has_value()) {
    return false;
  }
  if (!coordinator.try_reserve_model_memory(options_.model_id, *plan)) {
    pages.finish_resume(options_.model_id, /*success=*/false);
    LOG(ERROR) << "Insufficient shared capacity to wake model "
               << options_.model_id;
    return false;
  }
  pages.map_pages(options_.model_id, *plan);
  coordinator.restore_weight_pages(options_.model_id);
  pages.finish_resume(options_.model_id, /*success=*/true);

  LOG(INFO) << "Waking up model " << options_.model_id
            << ", remote_addrs.size()=" << options.remote_addrs.size();
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());

  if (!options.remote_addrs.empty() &&
      options.remote_addrs.size() == worker_clients.size()) {
    // Each worker pulls weights from the source at the same global rank.
    for (size_t i = 0; i < worker_clients.size(); ++i) {
      WakeupOptions per_worker_options;
      per_worker_options.master_status = options.master_status;
      per_worker_options.remote_addrs = {options.remote_addrs[i]};
      if (i < options.src_weight_segments.size()) {
        per_worker_options.src_weight_segments = {
            options.src_weight_segments[i]};
      }
      futures.emplace_back(worker_clients[i]->wakeup_async(per_worker_options));
    }
  } else {
    for (const auto& worker : worker_clients) {
      futures.emplace_back(worker->wakeup_async(options));
    }
  }

  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      LOG(ERROR) << "Wakeup failed.";
      return false;
    }
  }
  LOG(INFO) << "Wakeup finished for model " << options_.model_id << ".";
  return true;
}

bool ModelResidencyCoordinator::suspend_memory(bool skip_weight_release) {
  auto& coordinator = DistributedMemoryCoordinator::get_instance();
  auto& pages = coordinator.kv_cache_page_allocator();
  const auto plan = pages.begin_suspend(options_.model_id);
  if (!plan.has_value()) {
    return false;
  }
  if (!skip_weight_release) {
    coordinator.release_weight_pages(options_.model_id);
  }
  pages.unmap_pages(options_.model_id, *plan);
  coordinator.release_kv_mapping_budget(options_.model_id, *plan);
  pages.finish_suspend(options_.model_id);
  return true;
}

}  // namespace xllm

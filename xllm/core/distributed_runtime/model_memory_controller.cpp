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

#include "core/distributed_runtime/model_memory_controller.h"

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/allocator/model_memory_manager.h"
#include "core/framework/allocator/model_page_allocator.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_loader/model_loader.h"

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

  LOG(INFO) << "MappedMemoryRegion rolling_load weight budget: total="
            << all_size << ", non_decoder=" << non_decoder_size
            << ", all_decoder=" << all_decoder_size
            << ", max_layer=" << max_layer_size
            << ", rolling_buffer=" << rolling_buffer_size << " ("
            << LoadConfig::get_instance().rolling_load_num_cached_layers()
            << " slots x " << max_layer_size << " bytes/max-layer)"
            << ", effective=" << total_weight_size;
  return total_weight_size;
}

}  // namespace

ModelMemoryController::ModelMemoryController(
    Options options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(std::move(options)),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {}

bool ModelMemoryController::initialize_model(const ModelLoader& model_loader,
                                             int64_t num_layers,
                                             int32_t dp_size,
                                             int32_t tp_size,
                                             MasterStatus master_status) {
  if (!options_.enabled) {
    return true;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to initialize "
                  "MappedMemoryRegion model.";
    return false;
  }

  auto& page_allocator = ModelPageAllocator::get_instance();
  if (!page_allocator.is_initialized()) {
    auto& phy_pool = PhysicalPagePool::get_instance();
    CHECK(phy_pool.is_initialized())
        << "PhysicalPagePool must be initialized before ModelPageAllocator";
    const size_t num_phy_pages = phy_pool.num_total();
    const int32_t max_world_size = static_cast<int32_t>(
        distributed_worker_manager_->get_worker_clients().size());
    page_allocator.init(num_phy_pages,
                        dp_size,
                        max_world_size,
                        /*enable_page_prealloc=*/true);
  }

  // Each model owns its logical page list and shares the physical page pool.
  const std::string& model_id = options_.model_id;
  page_allocator.register_model(model_id, num_layers, master_status);
  page_allocator.set_model_parallel_strategy(model_id, dp_size, tp_size);
  auto& model_memory_manager = ModelMemoryManager::get_instance();
  model_memory_manager.set_model_parallel_strategy(model_id, dp_size, tp_size);

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

  LOG(INFO) << "MappedMemoryRegion weight allocation: total_weight_size="
            << total_weight_size << ", tp_size=" << tp_size
            << ", weight_size_per_tp=" << weight_size_per_tp
            << ", num_pages=" << num_pages
            << ", master_status=" << master_status;

  if (master_status == MasterStatus::WAKEUP) {
    if (!page_allocator.alloc_weight_pages(model_id, num_pages)) {
      LOG(ERROR) << "Failed to allocate weight pages";
      return false;
    }
    LOG(INFO)
        << "master_status=0 (MasterStatus::WAKEUP): Allocated weight pages, "
           "will load to device";
  } else if (master_status == MasterStatus::LIGHT_SLEEP ||
             master_status == MasterStatus::DEEP_SLEEP) {
    page_allocator.set_weight_pages_count(model_id, num_pages);
    LOG(INFO) << "master_status=" << master_status
              << " (SLEEP): Recorded weight pages, num_pages=" << num_pages;
  }

  return true;
}

bool ModelMemoryController::finish_initialization(MasterStatus master_status) {
  if (!options_.enabled || master_status == MasterStatus::WAKEUP) {
    return true;
  }

  // KV cache allocation must finish before releasing initial resources.
  if (!ModelPageAllocator::get_instance().sleep_model(
          options_.model_id, /*skip_weight_release=*/true)) {
    LOG(ERROR) << "Failed to sleep model " << options_.model_id
               << " after init";
    return false;
  }
  LOG(INFO) << "Model " << options_.model_id
            << " put to sleep after init (master_status=" << master_status
            << ")";
  return true;
}

void ModelMemoryController::get_virtual_memory_info(
    std::vector<size_t>& worker_free_phy_pages,
    std::unordered_map<std::string, std::vector<WeightSegment>>&
        model_weight_segments) const {
  if (!options_.enabled) {
    return;
  }

  // Worker 0 shares the master's process, so these queries need no RPC.
  auto& page_allocator = ModelPageAllocator::get_instance();
  if (page_allocator.is_initialized()) {
    worker_free_phy_pages = page_allocator.get_all_worker_free_pages();
  }

  auto& model_memory_manager = ModelMemoryManager::get_instance();
  model_weight_segments = model_memory_manager.get_all_model_weight_segments();
}

bool ModelMemoryController::sleep(MasterStatus master_status) {
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
  auto& page_allocator = ModelPageAllocator::get_instance();
  if (!page_allocator.sleep_model(options_.model_id)) {
    LOG(ERROR) << "ModelPageAllocator sleep_model failed, aborting sleep flow";
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

bool ModelMemoryController::wakeup(const WakeupOptions& options) {
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
  auto& page_allocator = ModelPageAllocator::get_instance();
  if (!page_allocator.wakeup_model(options_.model_id)) {
    LOG(ERROR)
        << "ModelPageAllocator wakeup_model failed, aborting wakeup flow";
    return false;
  }

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

bool ModelMemoryController::get_kv_cache_offsets_for_blocks(
    int32_t dp_rank,
    const std::vector<int32_t>& block_ids,
    uint64_t slot_size,
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
        layer_offsets) const {
  if (!options_.enabled) {
    return false;
  }

  const uint64_t block_size_bytes = slot_size * options_.block_size / 2;
  auto& allocator = ModelMemoryManager::get_instance();
  if (!allocator.get_kv_cache_offsets(dp_rank,
                                      options_.model_id,
                                      block_ids,
                                      block_size_bytes,
                                      layer_offsets)) {
    LOG(ERROR) << "get_kv_cache_offsets_for_blocks via RPC failed for dp_rank="
               << dp_rank << ", model_id=" << options_.model_id;
    return false;
  }

  VLOG(1) << "get_kv_cache_offsets_for_blocks: dp_rank=" << dp_rank
          << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();
  return true;
}

}  // namespace xllm

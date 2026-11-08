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

#include <algorithm>
#include <utility>
#include <vector>

#include "core/distributed_runtime/distributed_vmm_coordinator.h"
#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"
#include "core/runtime/worker_memory_resources.h"

namespace xllm {

namespace {

constexpr size_t kWeightPageSafetyMargin = 20;

bool is_sleep_status(MasterStatus status) {
  return status == MasterStatus::LIGHT_SLEEP ||
         status == MasterStatus::DEEP_SLEEP;
}

bool same_wakeup_options(const WakeupOptions& first,
                         const WakeupOptions& second) {
  if (first.master_status != second.master_status ||
      first.remote_addrs != second.remote_addrs ||
      first.src_weight_segments.size() != second.src_weight_segments.size()) {
    return false;
  }
  for (size_t rank = 0; rank < first.src_weight_segments.size(); ++rank) {
    const auto& first_segments = first.src_weight_segments[rank];
    const auto& second_segments = second.src_weight_segments[rank];
    if (first_segments.size() != second_segments.size() ||
        !std::equal(first_segments.begin(),
                    first_segments.end(),
                    second_segments.begin(),
                    [](const WeightSegment& lhs, const WeightSegment& rhs) {
                      return lhs.offset == rhs.offset && lhs.size == rhs.size;
                    })) {
      return false;
    }
  }
  return true;
}

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
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (!options_.enabled || !LoadConfig::get_instance().enable_vmm_weights()) {
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
  auto& coordinator = DistributedVmmCoordinator::get_instance();
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
        /*enable_prealloc=*/KVCacheConfig::get_instance().enable_vmm_kvcache());
  }

  const std::string& model_id = options_.model_id;
  coordinator.set_model_parallel_strategy(model_id, dp_size, tp_size);

  // Weight-only VMM still uses the residency transaction, with an empty KV
  // mapping plan. Physical KV tensors remain owned by their normal allocator.
  if (!KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      !coordinator.vmm_kv_page_allocator().register_model(model_id,
                                                          num_layers)) {
    return false;
  }

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
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (!options_.enabled || !LoadConfig::get_instance().enable_vmm_weights() ||
      master_status == MasterStatus::WAKEUP) {
    return true;
  }

  // KV cache allocation must finish before releasing initial resources.
  if (!suspend_memory(/*skip_weight_release=*/true)) {
    LOG(ERROR) << "Failed to sleep model " << options_.model_id
               << " after init";
    return false;
  }
  sleep_status_ = master_status;
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

  auto& coordinator = DistributedVmmCoordinator::get_instance();
  if (coordinator.is_initialized()) {
    worker_free_phy_pages = coordinator.get_worker_free_page_counts();
  }
  model_weight_segments =
      WorkerMemoryResources::get_instance().get_all_model_weight_segments();
}

bool ModelResidencyCoordinator::sleep(MasterStatus master_status) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (!options_.enabled || !LoadConfig::get_instance().enable_vmm_weights()) {
    LOG(WARNING) << "sleep requires --enable_vmm_weights=true";
    return false;
  }
  if (!is_sleep_status(master_status)) {
    LOG(ERROR) << "Invalid sleep status for model " << options_.model_id;
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to sleep.";
    return false;
  }
  if (pending_wakeup_.has_value()) {
    LOG(ERROR) << "Cannot sleep model " << options_.model_id
               << " while worker wakeup acknowledgements are pending";
    return false;
  }
  if (pending_sleep_.has_value()) {
    if (pending_sleep_->master_status != master_status) {
      LOG(ERROR) << "Cannot change the pending sleep status for model "
                 << options_.model_id;
      return false;
    }
    return finish_worker_sleep();
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

  ++transition_id_;
  sleep_status_ = master_status;
  pending_sleep_.emplace(PendingSleep{
      master_status, std::vector<bool>(worker_clients.size(), false)});
  return finish_worker_sleep();
}

bool ModelResidencyCoordinator::wakeup(const WakeupOptions& options) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (!options_.enabled || !LoadConfig::get_instance().enable_vmm_weights()) {
    LOG(WARNING) << "wakeup requires --enable_vmm_weights=true";
    return false;
  }
  if (!is_sleep_status(options.master_status) ||
      (sleep_status_.has_value() && *sleep_status_ != options.master_status)) {
    LOG(ERROR) << "Wakeup must match the sleep status for model "
               << options_.model_id;
    return false;
  }
  if (LoadConfig::get_instance().enable_vmm_weights() &&
      KVCacheConfig::get_instance().vmm_elastic() &&
      !options.remote_addrs.empty()) {
    // The current transport requires a dense, fully resident shared arena.
    // Reject this capability before reserving or restoring model resources.
    LOG(ERROR) << "Remote weight wakeup requires --vmm_elastic=false";
    return false;
  }
  if (distributed_worker_manager_ == nullptr ||
      distributed_worker_manager_->get_worker_clients().empty()) {
    LOG(ERROR) << "No worker clients available to wakeup.";
    return false;
  }

  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  if (pending_wakeup_.has_value() &&
      !same_wakeup_options(pending_wakeup_->options, options)) {
    LOG(ERROR) << "Cannot change the pending wakeup options for model "
               << options_.model_id;
    return false;
  }
  // A failed sleep may have left workers retaining different host state.
  // Complete that transition before restoring any shared device mappings.
  if (pending_sleep_.has_value() && !finish_worker_sleep()) {
    return false;
  }
  LOG(INFO) << "Starting to wakeup model " << options_.model_id
            << ". Worker clients count: " << worker_clients.size();

  // Restore weight and KV cache pages before workers load or transfer weights.
  auto& coordinator = DistributedVmmCoordinator::get_instance();
  auto& pages = coordinator.vmm_kv_page_allocator();
  if (!pending_wakeup_.has_value()) {
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
    if (!pages.map_pages(options_.model_id, *plan)) {
      coordinator.release_model_memory_budget(options_.model_id, *plan);
      pages.finish_resume(options_.model_id, /*success=*/false);
      LOG(ERROR)
          << "Insufficient physical memory to restore KV pages for model "
          << options_.model_id;
      return false;
    }
    if (!coordinator.restore_weight_pages(options_.model_id)) {
      // The weight transaction has already confirmed its rollback on every
      // worker. Release restored KV before returning the combined reservation.
      pages.unmap_pages(options_.model_id, *plan);
      coordinator.release_model_memory_budget(options_.model_id, *plan);
      pages.finish_resume(options_.model_id, /*success=*/false);
      LOG(ERROR) << "Insufficient physical memory to restore weights for model "
                 << options_.model_id;
      return false;
    }
    // Keep the allocator frozen in RESUMING while any worker can be using
    // these mappings. A retry reuses both the reservation and prepared pages.
    WakeupOptions requested_options = options;
    requested_options.transition_id = transition_id_;
    pending_wakeup_.emplace(
        PendingWakeup{std::move(requested_options),
                      std::vector<bool>(worker_clients.size(), false)});
  }

  LOG(INFO) << "Waking up model " << options_.model_id
            << ", remote_addrs.size()=" << options.remote_addrs.size();
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());
  std::vector<size_t> pending_ranks;
  pending_ranks.reserve(worker_clients.size());
  auto& pending = *pending_wakeup_;
  CHECK_EQ(pending.worker_acked.size(), worker_clients.size());
  const auto& prepared_options = pending.options;
  for (size_t rank = 0; rank < worker_clients.size(); ++rank) {
    if (pending.worker_acked[rank]) {
      continue;
    }
    pending_ranks.emplace_back(rank);
    if (!prepared_options.remote_addrs.empty() &&
        prepared_options.remote_addrs.size() == worker_clients.size()) {
      // Each worker pulls weights from the source at the same global rank.
      WakeupOptions per_worker_options;
      per_worker_options.master_status = prepared_options.master_status;
      per_worker_options.transition_id = prepared_options.transition_id;
      per_worker_options.remote_addrs = {prepared_options.remote_addrs[rank]};
      if (rank < prepared_options.src_weight_segments.size()) {
        per_worker_options.src_weight_segments = {
            prepared_options.src_weight_segments[rank]};
      }
      futures.emplace_back(
          worker_clients[rank]->wakeup_async(per_worker_options));
      continue;
    }
    futures.emplace_back(worker_clients[rank]->wakeup_async(prepared_options));
  }

  auto results = folly::collectAll(futures).get();
  bool all_acked = true;
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    const size_t rank = pending_ranks[index];
    if (result.hasValue() && result.value()) {
      pending.worker_acked[rank] = true;
      continue;
    }
    LOG(ERROR) << "Wakeup failed for model " << options_.model_id
               << " on worker " << rank;
    all_acked = false;
  }
  if (!all_acked) {
    return false;
  }
  pages.finish_resume(options_.model_id, /*success=*/true);
  pending_wakeup_.reset();
  LOG(INFO) << "Wakeup finished for model " << options_.model_id << ".";
  return true;
}

bool ModelResidencyCoordinator::is_suspended() const {
  if (!options_.enabled || !LoadConfig::get_instance().enable_vmm_weights()) {
    return false;
  }
  auto& coordinator = DistributedVmmCoordinator::get_instance();
  return coordinator.is_initialized() &&
         coordinator.vmm_kv_page_allocator().is_suspended(options_.model_id);
}

bool ModelResidencyCoordinator::finish_worker_sleep() {
  const auto& worker_clients =
      distributed_worker_manager_->get_worker_clients();
  auto& pending = *pending_sleep_;
  CHECK_EQ(pending.worker_acked.size(), worker_clients.size());
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients.size());
  std::vector<size_t> pending_ranks;
  pending_ranks.reserve(worker_clients.size());
  for (size_t rank = 0; rank < worker_clients.size(); ++rank) {
    if (pending.worker_acked[rank]) {
      continue;
    }
    pending_ranks.emplace_back(rank);
    futures.emplace_back(worker_clients[rank]->sleep_async(
        pending.master_status, transition_id_));
  }

  auto results = folly::collectAll(futures).get();
  bool all_acked = true;
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    const size_t rank = pending_ranks[index];
    if (result.hasValue() && result.value()) {
      pending.worker_acked[rank] = true;
      continue;
    }
    LOG(ERROR) << "Sleep failed for model " << options_.model_id
               << " on worker " << rank;
    all_acked = false;
  }
  if (all_acked) {
    pending_sleep_.reset();
  }
  return all_acked;
}

bool ModelResidencyCoordinator::suspend_memory(bool skip_weight_release) {
  auto& coordinator = DistributedVmmCoordinator::get_instance();
  auto& pages = coordinator.vmm_kv_page_allocator();
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

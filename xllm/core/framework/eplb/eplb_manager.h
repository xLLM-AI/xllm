/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include "core/framework/eplb/eplb_aggregator.h"
#include "core/framework/eplb/eplb_executor.h"
#include "core/framework/eplb/eplb_info.h"
#include "core/framework/eplb/eplb_options.h"
#include "core/framework/eplb/eplb_policy.h"
namespace xllm {

class EplbManager {
 public:
  // Initialize with model dimensions:
  // - layer_num: Total layers in the model
  // - device_num: Parallel devices in cluster
  // - experts_num: Experts per model layer
  EplbManager(int32_t layer_num, int32_t device_num, int32_t experts_num);

  EplbManager(int32_t layer_num,
              int32_t device_num,
              int32_t experts_num,
              EplbOptions options,
              std::unique_ptr<IEplbPolicy> eplb_policy);

  ~EplbManager();

  // Feed new expert workload data for load balancing
  // Input tensors should have shape [layer_num, experts_num]
  void update_expert_load(const std::vector<torch::Tensor>& expert_load,
                          int64_t completed_activation_token = -1);

  // Fetch current coordination instructions for expert updates. When commands
  // are not allowed, returns an empty result without advancing layer state.
  EplbInfo get_eplb_info(bool allow_eplb_command = true);

  // Mark specified layers as prepared (call after async loading completes)
  // prepare_tokens: Completed prepare-attempt tokens per device.
  void set_prepared_tokens(const std::vector<int64_t>& prepare_tokens);

 private:
  // Thread functions
  void rebalance_experts_loop();
  void eplb_manager_loop();
  // Return the index of the next `true` entry in `vec` at or after
  // `start_pos`, or -1 if none. -1 is the shared "no active layer" sentinel
  // used by state_.active_layer_id.
  int32_t find_next_true(const std::vector<bool>& vec, size_t start_pos);

  // Per-layer preparation lifecycle. Separates the (int) layer id being
  // advanced by rebalance_experts_loop from the (state) each layer is in, so
  // "-1" no longer has to mean both "no active layer" and "field invalid".
  //   IDLE      -> layer never selected in this round
  //   PREPARING -> get_eplb_info handed the layer to workers; waiting on their
  //                set_prepared_tokens callbacks
  //   READY     -> every device reported prepared; awaits get_eplb_info to
  //                report update_layer_id
  enum class LayerState : int8_t { IDLE, PREPARING, READY };

  struct ExpertLoadSample {
    std::vector<torch::Tensor> expert_loads;
    torch::Tensor active_expert_distribution;
    int64_t active_distribution_generation = 0;
  };

  // Shared data with mutex protection.
  //
  // Concurrency contract:
  //   - Every non-const field below MUST be read or written with `mtx` held
  //     by any thread other than the constructor's setup phase (which runs
  //     before rebalance_thread_ / manager_thread_ are spawned).
  //   - `data_cv` is signalled by producers of expert_load_queue
  //     (update_expert_load) and awaited by the rebalance thread.
  //   - `state_cv` is signalled whenever a layer's lifecycle field
  //     (active_layer_id / prepared_tokens / layer_states) advances, and
  //     awaited by the manager thread + destructor.
  //   - Long-running work (aggregation, policy rebalance, host copies) runs
  //     on a *local snapshot* taken under the mutex so the hot path
  //     (update_expert_load / get_eplb_info) does not stall.
  //   - refresh_cached_expert_ids() writes cached_expert_ids and must be
  //     called with the mutex held (see the assertion comment on that method).
  struct ThreadSafeData {
    std::mutex mtx;
    std::condition_variable data_cv;
    std::condition_variable state_cv;
    bool stop = false;

    // Expert load tracking
    torch::Tensor expert_load;
    // Logical and physical load are scoped to the same active placement
    // generation. Both reset together when a layer activation commits.
    // Measured physical slot load has shape [layer, device, slot].
    torch::Tensor physical_expert_load;
    // Placement currently used by model forwards. This tensor is never
    // mutated in place after publication: activation clones it and swaps in a
    // new tensor, so queued samples can retain a cheap immutable snapshot.
    torch::Tensor active_expert_distribution;
    // Generation of active_expert_distribution. Physical load is valid only
    // for samples captured from this generation; activation increments it and
    // starts a fresh physical workload window.
    int64_t active_distribution_generation = 0;
    // Number of samples aggregated under active_distribution_generation.
    // Policy evaluation is deferred while this is zero because a zero-filled
    // tensor is not a measured physical baseline.
    int64_t physical_load_sample_count = 0;
    // Placement proposed by the latest policy round and handed to prepare.
    torch::Tensor expert_distribution;
    std::vector<bool> enable_update_vec;
    std::queue<ExpertLoadSample> expert_load_queue;

    // Layer state tracking
    std::vector<LayerState> layer_states;
    std::vector<int64_t> prepared_tokens;
    int64_t active_prepare_token = -1;
    int64_t next_prepare_token = 1;
    std::optional<std::chrono::steady_clock::time_point> prepare_dispatch_start;
    std::optional<std::chrono::steady_clock::time_point>
        prepare_observation_start;
    // Layer id currently being advanced by rebalance_experts_loop. -1 means
    // no layer is queued for prepare in this round.
    int32_t active_layer_id = -1;
    // Layer id waiting for get_eplb_info to publish as update_layer_id. -1
    // means nothing is pending activation.
    int32_t pending_activation_layer_id = -1;
    int64_t pending_activation_token = -1;
    // Layer whose activation command has been sent to workers. The engine
    // acknowledges it only when that command's worker output is collected.
    int32_t activation_awaiting_load_sample_layer_id = -1;
    int64_t activation_awaiting_completion_token = -1;

    // Host-resident cache of the current rebalance round's per-layer expert
    // distribution. Stored as a single flat [layer_num * device_num *
    // device_experts_num] int32 buffer for cache-friendly access; slices are
    // handed out by get_eplb_info() as `vector<int32_t>` copies over
    // [layer_stride * layer, layer_stride * (layer+1)). Populated
    // incrementally by rebalance_experts_loop -- only the layers that flipped
    // in enable_update_vec are refreshed each round, so an untouched layer
    // keeps its previous host copy without a device->host tensor round trip.
    // Size == layer_num_ * layer_stride_ once initialized.
    std::vector<int32_t> cached_expert_ids;
  };

  // Components: config-derived scalars are declared first so aggregator_
  // (which needs layer_num_/device_num_/device_experts_num_) and the state
  // torch tensors (which need layer_num_/device_num_/device_experts_num_
  // during their initialization) can be constructed after them without
  // relying on unspecified initialization order.
  // Snapshot of the EPLB tunables at construction time. Every thread inside
  // this manager reads from `options_` instead of reaching back into the
  // process-wide EPLBConfig::get_instance(), so a single manager sees a
  // stable configuration and unit tests can construct one with a custom
  // EplbOptions without mutating global state.
  EplbOptions options_;

  // Constants
  const int32_t layer_num_;
  const int32_t device_num_;
  const int32_t experts_num_;
  const int32_t device_experts_num_;
  // Flat stride (device_num_ * device_experts_num_) between layers inside
  // state_.cached_expert_ids. Computed once at construction.
  const int64_t layer_stride_;

  std::unique_ptr<IEplbPolicy> eplb_policy_ = nullptr;
  // Stateless helper that folds per-device cumulative expert counters into
  // the manager's global load tensor. Extracted from EplbManager so it can
  // be exercised in isolation and swapped for a mock in tests.
  EplbAggregator aggregator_;
  ThreadSafeData state_;

  // Threads
  std::thread rebalance_thread_;
  std::thread manager_thread_;

  // Materialize per-layer int32 flat views of expert_distribution into
  // state_.cached_expert_ids. Runs once per rebalance under the state mutex.
  // If `layers_to_refresh` is empty, every layer is copied (used at boot);
  // otherwise only the listed layer ids are refreshed, so untouched layers
  // keep their previous host copy and pay no device->host round trip.
  void refresh_cached_expert_ids(
      const std::vector<int32_t>& layers_to_refresh = {});
};

}  // namespace xllm

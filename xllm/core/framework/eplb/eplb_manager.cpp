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

#include "core/framework/eplb/eplb_manager.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "core/framework/eplb/eplb_utils.h"

namespace xllm {

namespace {

EplbOptions validated_options(EplbOptions options) {
  options.validate();
  return options;
}

}  // namespace

using namespace std::chrono_literals;

EplbManager::EplbManager(int32_t layer_num,
                         int32_t device_num,
                         int32_t experts_num)
    : EplbManager(layer_num,
                  device_num,
                  experts_num,
                  EplbOptions::from_global_config(),
                  nullptr) {}

EplbManager::EplbManager(int32_t layer_num,
                         int32_t device_num,
                         int32_t experts_num,
                         EplbOptions options,
                         std::unique_ptr<IEplbPolicy> eplb_policy)
    : options_(validated_options(std::move(options))),
      layer_num_(layer_num),
      device_num_(device_num),
      experts_num_(experts_num),
      device_experts_num_(
          eplb::local_physical_experts_num(experts_num,
                                           device_num,
                                           options_.redundant_experts_num)),
      layer_stride_(static_cast<int64_t>(device_num) * device_experts_num_),
      eplb_policy_(std::move(eplb_policy)),
      aggregator_(layer_num, device_num, device_experts_num_) {
  // No other thread has been spawned yet, so state_ is single-writer and
  // does not need the mutex to be held here. The mutex is initialized to
  // "unlocked" per the standard, and the rebalance / manager threads below
  // acquire it before their first read.
  if (eplb_policy_ == nullptr) {
    const EplbPolicyKind policy_kind =
        eplb_policy_kind_from_string(options_.eplb_policy_kind);
    eplb_policy_ = MakeEplbPolicy(
        policy_kind, device_experts_num_, device_num_, layer_num_, options_);
  }
  LOG(INFO) << "EPLB manager start | policy=" << eplb_policy_->name()
            << " | layers=" << layer_num_ << " | devices=" << device_num_
            << " | device_experts=" << device_experts_num_;
  state_.expert_load = torch::zeros({layer_num_, experts_num_}, torch::kInt64);
  state_.physical_expert_load = torch::zeros(
      {layer_num_, device_num_, device_experts_num_}, torch::kInt64);
  state_.prepared_tokens.resize(device_num, -1);
  state_.layer_states.assign(layer_num_, LayerState::IDLE);
  state_.expert_distribution = torch::zeros(
      {layer_num_, device_num_, device_experts_num_}, torch::kInt32);
  for (int32_t layer = 0; layer < layer_num_; ++layer) {
    for (int32_t device = 0; device < device_num_; ++device) {
      int32_t device_route_experts_num =
          device_experts_num_ - options_.redundant_experts_num;
      int32_t base = device * device_route_experts_num;
      for (int32_t expert = 0; expert < device_experts_num_; ++expert) {
        int32_t value = base + expert;
        if (expert >= device_route_experts_num) {
          value = base + device_route_experts_num - 1;
        }
        state_.expert_distribution[layer][device][expert] = value;
      }
    }
  }
  state_.active_expert_distribution = state_.expert_distribution.clone();
  eplb_policy_->initialize_distribution(state_.active_expert_distribution);
  refresh_cached_expert_ids();

  // Start worker threads
  rebalance_thread_ = std::thread(&EplbManager::rebalance_experts_loop, this);
  manager_thread_ = std::thread(&EplbManager::eplb_manager_loop, this);
}

EplbManager::~EplbManager() {
  {
    std::lock_guard<std::mutex> lock(state_.mtx);
    state_.stop = true;
    state_.data_cv.notify_all();
    state_.state_cv.notify_all();
  }

  if (rebalance_thread_.joinable()) {
    rebalance_thread_.join();
  }
  if (manager_thread_.joinable()) {
    manager_thread_.join();
  }
}

void EplbManager::update_expert_load(
    const std::vector<torch::Tensor>& expert_load,
    int64_t completed_activation_token) {
  CHECK_EQ(expert_load.size(), static_cast<size_t>(device_num_))
      << "EPLB expert load tensor count must match device_num.";
  std::lock_guard<std::mutex> lock(state_.mtx);
  state_.expert_load_queue.push(
      ExpertLoadSample{expert_load,
                       state_.active_expert_distribution,
                       state_.active_distribution_generation});
  if (state_.activation_awaiting_load_sample_layer_id != -1 &&
      completed_activation_token ==
          state_.activation_awaiting_completion_token) {
    const int32_t activated_layer =
        state_.activation_awaiting_load_sample_layer_id;
    eplb_policy_->commit_layer(activated_layer);
    torch::Tensor next_active = state_.active_expert_distribution.clone();
    next_active[activated_layer].copy_(
        state_.expert_distribution[activated_layer]);
    state_.active_expert_distribution = std::move(next_active);
    ++state_.active_distribution_generation;
    state_.expert_load.zero_();
    state_.physical_expert_load.zero_();
    state_.physical_load_sample_count = 0;
    state_.activation_awaiting_load_sample_layer_id = -1;
    state_.activation_awaiting_completion_token = -1;
  } else if (completed_activation_token != -1) {
    LOG(WARNING) << "Ignoring stale EPLB activation completion token "
                 << completed_activation_token << "; waiting for "
                 << state_.activation_awaiting_completion_token;
  }
  state_.data_cv.notify_one();
}

void EplbManager::rebalance_experts_loop() {
  std::optional<std::chrono::steady_clock::time_point> rebalance_deadline;
  int64_t deadline_generation = -1;
  int64_t heartbeat_last_time = absl::ToUnixSeconds(absl::Now());
  int64_t heartbeat_batches = 0;
  int64_t heartbeat_updates = 0;
  int64_t last_rebalance_ms = 0;
  while (true) {
    std::vector<ExpertLoadSample> next_batch;
    torch::Tensor expert_load_snapshot;
    torch::Tensor physical_expert_load_snapshot;
    int64_t physical_load_generation = 0;
    int64_t physical_load_sample_count = 0;
    bool round_in_progress = false;
    // Phase 1: quickly grab whatever is waiting in the queue and release the
    // mutex, so update_expert_load / get_eplb_info from the engine hot path do
    // not stall behind aggregation and policy evaluation.
    {
      std::unique_lock<std::mutex> lock(state_.mtx);
      const auto has_data = [&] {
        return state_.stop || !state_.expert_load_queue.empty();
      };
      if (rebalance_deadline.has_value()) {
        state_.data_cv.wait_until(lock, rebalance_deadline.value(), has_data);
      } else {
        state_.data_cv.wait(lock, has_data);
      }
      if (state_.stop) {
        return;
      }
      next_batch.reserve(state_.expert_load_queue.size());
      while (!state_.expert_load_queue.empty()) {
        next_batch.emplace_back(std::move(state_.expert_load_queue.front()));
        state_.expert_load_queue.pop();
      }
      expert_load_snapshot = state_.expert_load.clone();
      physical_expert_load_snapshot = state_.physical_expert_load.clone();
      physical_load_generation = state_.active_distribution_generation;
      physical_load_sample_count = state_.physical_load_sample_count;
      round_in_progress = state_.active_layer_id != -1 ||
                          state_.pending_activation_layer_id != -1 ||
                          state_.activation_awaiting_load_sample_layer_id != -1;
    }

    if (deadline_generation != physical_load_generation) {
      rebalance_deadline.reset();
      deadline_generation = physical_load_generation;
    }

    // Phase 2: aggregation runs on our local snapshots without holding mtx.
    for (ExpertLoadSample& sample : next_batch) {
      if (sample.active_distribution_generation == physical_load_generation) {
        aggregator_.aggregate(expert_load_snapshot,
                              physical_expert_load_snapshot,
                              sample.active_expert_distribution,
                              sample.expert_loads);
        ++physical_load_sample_count;
      }
    }

    // Phase 3: only run the (expensive) policy if the update interval elapsed.
    const auto now = std::chrono::steady_clock::now();
    if (physical_load_sample_count > 0 && !rebalance_deadline.has_value()) {
      rebalance_deadline =
          now + std::chrono::seconds(options_.eplb_update_interval);
    }
    int64_t current_time = absl::ToUnixSeconds(absl::Now());
    ++heartbeat_batches;
    bool rebalance_ready = false;
    torch::Tensor new_distribution;
    std::vector<bool> new_update_vec;
    int64_t updated_layer_count = 0;
    const bool update_interval_elapsed =
        rebalance_deadline.has_value() && now >= rebalance_deadline.value();
    if (!round_in_progress && physical_load_sample_count > 0 &&
        update_interval_elapsed) {
      rebalance_deadline.reset();
      auto rebalance_start = std::chrono::steady_clock::now();
      auto result = eplb_policy_->rebalance_experts(
          expert_load_snapshot, physical_expert_load_snapshot);
      new_distribution = std::move(result.first);
      new_update_vec = std::move(result.second);
      auto rebalance_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - rebalance_start)
              .count();
      last_rebalance_ms = rebalance_ms;
      for (bool update : new_update_vec) {
        if (update) {
          ++updated_layer_count;
        }
      }
      heartbeat_updates += updated_layer_count;
      LOG(INFO) << "EPLB rebalance | policy=" << eplb_policy_->name()
                << " | layers_updated=" << updated_layer_count << "/"
                << new_update_vec.size() << " | duration=" << rebalance_ms
                << "ms";
      rebalance_ready = true;
    } else if (update_interval_elapsed) {
      rebalance_deadline.reset();
    }

    // 60s heartbeat so on-call has a live signal of the rebalance thread even
    // when no layer passes the peak-load benefit gate in a whole minute.
    if (current_time - heartbeat_last_time >= 60) {
      LOG(INFO) << "EPLB heartbeat | rebalance_thread | policy="
                << eplb_policy_->name()
                << " | batches_since_last=" << heartbeat_batches
                << " | layers_updated_since_last=" << heartbeat_updates
                << " | last_rebalance_ms=" << last_rebalance_ms;
      heartbeat_last_time = current_time;
      heartbeat_batches = 0;
      heartbeat_updates = 0;
    }

    // Phase 4: publish results under the mutex; this window is O(1)/O(layers).
    std::vector<int32_t> layers_to_refresh;
    if (rebalance_ready) {
      layers_to_refresh.reserve(new_update_vec.size());
      for (int32_t layer = 0;
           layer < static_cast<int32_t>(new_update_vec.size());
           ++layer) {
        if (new_update_vec[layer]) {
          layers_to_refresh.push_back(layer);
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(state_.mtx);
      if (state_.active_distribution_generation == physical_load_generation) {
        state_.expert_load = expert_load_snapshot;
        state_.physical_expert_load = physical_expert_load_snapshot;
        state_.physical_load_sample_count = physical_load_sample_count;
      }
      if (rebalance_ready) {
        // Only copy per-layer slices for layers the policy actually rebalanced
        // this round. Non-triggered layers keep whatever plan they published
        // last, so a policy that leaves their row as its -1 sentinel does not
        // wipe the identity mapping installed by the manager constructor.
        for (int32_t layer : layers_to_refresh) {
          state_.expert_distribution[layer].copy_(new_distribution[layer]);
        }
        state_.enable_update_vec = std::move(new_update_vec);
        // Keep a short history window without exposing another operator
        // setting. This is the stable decay used by the original EPLB path.
        constexpr double kLoadDecay = 0.5;
        state_.expert_load =
            (state_.expert_load.to(torch::kFloat64) * kLoadDecay)
                .to(torch::kInt64);
        state_.physical_expert_load =
            (state_.physical_expert_load.to(torch::kFloat64) * kLoadDecay)
                .to(torch::kInt64);
        // Fresh round: reset per-layer lifecycle and pick the first layer the
        // policy actually wants to update.
        std::fill(state_.layer_states.begin(),
                  state_.layer_states.end(),
                  LayerState::IDLE);
        state_.pending_activation_layer_id = -1;
        state_.active_layer_id = find_next_true(state_.enable_update_vec, 0);
        // Refresh only the layers that just changed. Untouched layers keep
        // their previous host copy and skip the device->host round trip.
        refresh_cached_expert_ids(layers_to_refresh);
        state_.state_cv.notify_all();
      }
    }
  }
}

int32_t EplbManager::find_next_true(const std::vector<bool>& vec,
                                    size_t start_pos) {
  if (start_pos >= vec.size()) {
    return -1;
  }
  auto begin = vec.begin() + start_pos;
  auto it = std::find(begin, vec.end(), true);
  return (it != vec.end()) ? static_cast<int32_t>(it - vec.begin()) : -1;
}

void EplbManager::eplb_manager_loop() {
  // Backstop for a prepare attempt that never completes. Unique prepare tokens
  // let us safely skip the attempt: any late completion carries the old token
  // and cannot satisfy a later retry of the same layer.
  const auto kPrepareTimeout =
      std::chrono::seconds(options_.eplb_prepare_timeout_seconds);
  const auto kPrepareOutputTimeout = kPrepareTimeout * 2;
  CHECK_GT(options_.eplb_prepare_timeout_seconds, 0)
      << "eplb_prepare_timeout_seconds must be positive; got "
      << options_.eplb_prepare_timeout_seconds;
  auto heartbeat_last = std::chrono::steady_clock::now();
  int64_t heartbeat_layers_completed = 0;
  int64_t heartbeat_layers_timed_out = 0;
  const auto abort_active_prepare = [&](const char* timeout_kind,
                                        std::chrono::seconds timeout) {
    const int32_t timed_out_layer = state_.active_layer_id;
    LOG(WARNING) << "EPLB layer " << timed_out_layer << " " << timeout_kind
                 << " timed out after " << timeout.count()
                 << "s; skipping this layer for the current round.";
    eplb_policy_->abort_layer(timed_out_layer);
    state_.enable_update_vec[static_cast<size_t>(timed_out_layer)] = false;
    state_.layer_states[static_cast<size_t>(timed_out_layer)] =
        LayerState::IDLE;
    state_.active_prepare_token = -1;
    state_.prepare_dispatch_start.reset();
    state_.prepare_observation_start.reset();
    std::fill(state_.prepared_tokens.begin(), state_.prepared_tokens.end(), -1);
    state_.active_layer_id = find_next_true(
        state_.enable_update_vec, static_cast<size_t>(timed_out_layer) + 1);
    state_.state_cv.notify_all();
    ++heartbeat_layers_timed_out;
  };
  while (true) {
    {
      std::unique_lock<std::mutex> lock(state_.mtx);
      state_.state_cv.wait(
          lock, [&] { return state_.active_layer_id != -1 || state_.stop; });

      if (state_.stop) {
        return;
      }
    }
    while (true) {
      std::unique_lock<std::mutex> lock(state_.mtx);
      if (state_.stop) {
        return;
      }
      if (state_.active_layer_id < 0) {
        break;
      }
      if (state_.active_prepare_token == -1) {
        state_.state_cv.wait_for(lock, 10ms, [&] {
          return state_.stop || state_.active_prepare_token != -1;
        });
        continue;
      }
      bool all_prepared = true;
      for (int64_t prepare_token : state_.prepared_tokens) {
        if (prepare_token != state_.active_prepare_token) {
          all_prepared = false;
          break;
        }
      }
      if (all_prepared) {
        state_.layer_states[state_.active_layer_id] = LayerState::READY;
        state_.pending_activation_layer_id = state_.active_layer_id;
        state_.pending_activation_token = state_.active_prepare_token;
        state_.active_prepare_token = -1;
        state_.prepare_dispatch_start.reset();
        state_.prepare_observation_start.reset();
        state_.active_layer_id =
            find_next_true(state_.enable_update_vec,
                           static_cast<size_t>(state_.active_layer_id) + 1);
        ++heartbeat_layers_completed;
        continue;
      }
      // Not all devices have reported this layer prepared yet. Block until a
      // device reports progress (set_prepared_tokens) or shutdown instead of
      // busy-waiting and pinning a CPU core.
      state_.state_cv.wait_for(lock, 10ms);
      const std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now();
      if (!state_.prepare_observation_start.has_value()) {
        if (state_.prepare_dispatch_start.has_value() &&
            now - state_.prepare_dispatch_start.value() >
                kPrepareOutputTimeout) {
          abort_active_prepare("output", kPrepareOutputTimeout);
        }
        continue;
      }
      if (now - state_.prepare_observation_start.value() > kPrepareTimeout) {
        abort_active_prepare("prepare progress", kPrepareTimeout);
        continue;
      }
      if (std::chrono::steady_clock::now() - heartbeat_last >=
          std::chrono::seconds(60)) {
        LOG(INFO) << "EPLB heartbeat | manager_thread | layers_completed="
                  << heartbeat_layers_completed
                  << " | layers_timed_out=" << heartbeat_layers_timed_out
                  << " | waiting_on=" << state_.active_layer_id;
        heartbeat_last = std::chrono::steady_clock::now();
        heartbeat_layers_completed = 0;
        heartbeat_layers_timed_out = 0;
      }
    }
  }
}

EplbInfo EplbManager::get_eplb_info(bool allow_eplb_command) {
  EplbInfo info;
  if (!allow_eplb_command) {
    return info;
  }
  {
    std::lock_guard<std::mutex> lock(state_.mtx);
    if (state_.activation_awaiting_load_sample_layer_id == -1) {
      info.update_layer_id = state_.pending_activation_layer_id;
    }
    if (info.update_layer_id != -1) {
      CHECK_GT(state_.pending_activation_token, 0);
      info.activation_token = state_.pending_activation_token;
      state_.activation_awaiting_load_sample_layer_id = info.update_layer_id;
      state_.activation_awaiting_completion_token = info.activation_token;
      state_.pending_activation_layer_id = -1;
      state_.pending_activation_token = -1;
    }
    const int32_t active = state_.active_layer_id;
    if (active != -1 && state_.layer_states[active] == LayerState::IDLE) {
      info.prepare_layer_id = active;
      CHECK_GT(state_.next_prepare_token, 0);
      CHECK_LT(state_.next_prepare_token, std::numeric_limits<int64_t>::max())
          << "EPLB prepare token space exhausted.";
      info.prepare_token = state_.next_prepare_token;
      ++state_.next_prepare_token;
      state_.active_prepare_token = info.prepare_token;
      state_.prepare_dispatch_start = std::chrono::steady_clock::now();
      state_.prepare_observation_start.reset();
      // Read the pre-materialized host-side flat view instead of slicing
      // expert_distribution + .contiguous() + data_ptr copy on every step.
      // The cache was populated once by refresh_cached_expert_ids() when
      // rebalance_experts_loop published this round's distribution.
      const int64_t begin = static_cast<int64_t>(active) * layer_stride_;
      const int64_t end = begin + layer_stride_;
      CHECK_LE(static_cast<size_t>(end), state_.cached_expert_ids.size())
          << "cached_expert_ids too small for active layer " << active
          << " (need " << end << ", have " << state_.cached_expert_ids.size()
          << ")";
      info.expert_ids.assign(state_.cached_expert_ids.begin() + begin,
                             state_.cached_expert_ids.begin() + end);
      state_.layer_states[active] = LayerState::PREPARING;
      state_.state_cv.notify_all();
    } else {
      info.prepare_layer_id = -1;
    }
  }
  return info;
}

void EplbManager::refresh_cached_expert_ids(
    const std::vector<int32_t>& layers_to_refresh) {
  // Must be called with state_.mtx held (or before threads start). Rebuilds
  // the host-side flat view of expert_distribution so subsequent
  // get_eplb_info calls can hand out a copy in O(layer_stride_) without
  // touching torch tensors. Passing an empty layers_to_refresh means "refresh
  // all layers" (used at boot); otherwise only the listed layers are copied,
  // so untouched layers keep the previous host bytes.
  CHECK(state_.expert_distribution.defined())
      << "expert_distribution must be initialized before refresh.";
  CHECK_EQ(state_.expert_distribution.dtype(), torch::kInt32)
      << "expert_distribution must be int32; got "
      << state_.expert_distribution.dtype();
  const size_t total_slots =
      static_cast<size_t>(layer_num_) * static_cast<size_t>(layer_stride_);
  if (state_.cached_expert_ids.size() != total_slots) {
    state_.cached_expert_ids.assign(total_slots, 0);
  }
  if (layers_to_refresh.empty()) {
    const torch::Tensor host_view =
        state_.expert_distribution.to(torch::kCPU).contiguous();
    const int32_t* data = host_view.data_ptr<int32_t>();
    std::copy(data, data + total_slots, state_.cached_expert_ids.begin());
    return;
  }
  for (int32_t layer : layers_to_refresh) {
    CHECK(layer >= 0 && layer < layer_num_)
        << "refresh_cached_expert_ids layer id out of range: " << layer;
    const torch::Tensor host_slice =
        state_.expert_distribution[layer].to(torch::kCPU).contiguous();
    const int32_t* data = host_slice.data_ptr<int32_t>();
    const int64_t begin = static_cast<int64_t>(layer) * layer_stride_;
    std::copy(
        data, data + layer_stride_, state_.cached_expert_ids.begin() + begin);
  }
}

void EplbManager::set_prepared_tokens(
    const std::vector<int64_t>& prepare_tokens) {
  CHECK_EQ(prepare_tokens.size(), static_cast<size_t>(device_num_))
      << "EPLB prepare token count must match device_num.";
  std::lock_guard<std::mutex> lock(state_.mtx);
  if (state_.active_prepare_token != -1 &&
      !state_.prepare_observation_start.has_value()) {
    state_.prepare_observation_start = std::chrono::steady_clock::now();
  }
  for (size_t i = 0;
       i < prepare_tokens.size() && i < state_.prepared_tokens.size();
       ++i) {
    if (prepare_tokens[i] == state_.active_prepare_token) {
      state_.prepared_tokens[i] = prepare_tokens[i];
    }
  }
  state_.state_cv.notify_all();
}

}  // namespace xllm

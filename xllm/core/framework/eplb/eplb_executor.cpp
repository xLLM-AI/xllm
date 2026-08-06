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

#include "core/framework/eplb/eplb_executor.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "runtime/forward_params.h"

namespace xllm {

EplbExecutor::EplbExecutor(CausalLM& model, const torch::Device& device)
    : model_(model),
      device_(device),
      stream_(device_.get_stream_from_pool()),
      eplb_worker_(&EplbExecutor::eplb_worker_loop, this) {}

EplbExecutor::~EplbExecutor() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_one();
  if (eplb_worker_.joinable()) {
    eplb_worker_.join();
  }
}

void EplbExecutor::eplb_execute(const EplbInfo& eplb_info) {
  start_eplb_step(eplb_info);
  finish_eplb_step();
}

void EplbExecutor::start_eplb_step(const EplbInfo& eplb_info) {
  std::optional<Task> prepare_task;
  if (eplb_info.prepare_layer_id != -1) {
    CHECK_GT(eplb_info.prepare_token, 0)
        << "EPLB prepare command requires a positive attempt token.";
    prepare_task.emplace(
        Task{eplb_info.prepare_layer_id,
             eplb_info.prepare_token,
             eplb_info.expert_ids,
             [prepare_layer_id = eplb_info.prepare_layer_id](int32_t) {
               LOG(INFO) << "prepare expert weight complete, layer: "
                         << prepare_layer_id << std::endl;
             }});
  }
  {
    std::lock_guard<std::mutex> lock(update_mutex_);
    if (eplb_info.update_layer_id != -1) {
      CHECK_EQ(pending_update_layer_id_, -1)
          << "Cannot start a new EPLB transfer before the previous step is "
             "finished.";
      model_.start_expert_weight_transfer(eplb_info.update_layer_id);
      pending_update_layer_id_ = eplb_info.update_layer_id;
    }
    if (prepare_task.has_value() && pending_update_layer_id_ != -1) {
      CHECK(!deferred_prepare_task_.has_value())
          << "Cannot defer more than one EPLB prepare task per update step.";
      deferred_prepare_task_ = std::move(*prepare_task);
      prepare_task.reset();
    }
  }
  if (prepare_task.has_value()) {
    enqueue_prepare_task(std::move(*prepare_task));
  }
}

void EplbExecutor::finish_eplb_step() {
  std::optional<Task> deferred_prepare_task;
  {
    std::lock_guard<std::mutex> lock(update_mutex_);
    if (pending_update_layer_id_ == -1) {
      CHECK(!deferred_prepare_task_.has_value())
          << "Deferred EPLB prepare task has no pending update.";
      return;
    }
    model_.update_expert_weight(pending_update_layer_id_);
    pending_update_layer_id_ = -1;
    deferred_prepare_task = std::move(deferred_prepare_task_);
    deferred_prepare_task_.reset();
  }
  if (deferred_prepare_task.has_value()) {
    enqueue_prepare_task(std::move(*deferred_prepare_task));
  }
}

void EplbExecutor::enqueue_prepare_task(Task task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_.emplace(std::move(task));
  }
  condition_.notify_one();
}

void EplbExecutor::prepare_expert_weight_async(
    int32_t layer_id,
    int64_t prepare_token,
    const std::vector<int32_t>& expert_ids,
    Callback callback) {
  enqueue_prepare_task(Task{layer_id, prepare_token, expert_ids, callback});
}

int64_t EplbExecutor::consume_ready_prepare_token() {
  std::lock_guard<std::mutex> lock(ready_mutex_);
  const int64_t ready_prepare_token = ready_prepare_token_;
  ready_prepare_token_ = -1;
  return ready_prepare_token;
}

void EplbExecutor::eplb_worker_loop() {
  auto heartbeat_last = std::chrono::steady_clock::now();
  int64_t heartbeat_tasks_ok = 0;
  int64_t heartbeat_tasks_failed = 0;
  int64_t last_prepare_ms = 0;
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      condition_.wait_for(lock, std::chrono::seconds(60), [this] {
        return !tasks_.empty() || stop_;
      });
      if (stop_) {
        return;
      }
      if (tasks_.empty()) {
        // Heartbeat tick while nothing is queued so on-call has a live signal
        // that this thread is up and idle rather than deadlocked.
        if (std::chrono::steady_clock::now() - heartbeat_last >=
            std::chrono::seconds(60)) {
          LOG(INFO) << "EPLB heartbeat | executor_thread | tasks_ok_since_last="
                    << heartbeat_tasks_ok
                    << " | tasks_failed_since_last=" << heartbeat_tasks_failed
                    << " | last_prepare_ms=" << last_prepare_ms
                    << " | queued=0";
          heartbeat_last = std::chrono::steady_clock::now();
          heartbeat_tasks_ok = 0;
          heartbeat_tasks_failed = 0;
        }
        continue;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    auto prepare_start = std::chrono::high_resolution_clock::now();

    c10::StreamGuard streamGuard = stream_->set_stream_guard();
    model_.prepare_expert_weight(task.layer_id, task.expert_ids);
    const int32_t synchronize_result = stream_->synchronize();
    auto prepare_end = std::chrono::high_resolution_clock::now();
    auto prepare_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(prepare_end -
                                                              prepare_start)
            .count();
    last_prepare_ms = prepare_duration;
    LOG(INFO) << "prepare_expert_weight | layer=" << task.layer_id
              << " | experts=" << task.expert_ids.size()
              << " | duration=" << prepare_duration << "ms";
    const bool prepare_succeeded =
        synchronize_result == 0 &&
        model_.last_prepare_expert_weight_ok(task.layer_id);
    if (prepare_succeeded) {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      ready_prepare_token_ = task.prepare_token;
      ++heartbeat_tasks_ok;
    } else {
      LOG(ERROR) << "prepare_expert_weight failed for layer " << task.layer_id
                 << " | stream_synchronize_result=" << synchronize_result
                 << "; not advancing ready_layer_id.";
      ++heartbeat_tasks_failed;
    }
    if (prepare_succeeded && task.callback) {
      task.callback(task.layer_id);
    }
    if (std::chrono::steady_clock::now() - heartbeat_last >=
        std::chrono::seconds(60)) {
      LOG(INFO) << "EPLB heartbeat | executor_thread | tasks_ok_since_last="
                << heartbeat_tasks_ok
                << " | tasks_failed_since_last=" << heartbeat_tasks_failed
                << " | last_prepare_ms=" << last_prepare_ms;
      heartbeat_last = std::chrono::steady_clock::now();
      heartbeat_tasks_ok = 0;
      heartbeat_tasks_failed = 0;
    }
  }
}
}  // namespace xllm

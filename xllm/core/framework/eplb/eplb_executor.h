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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#include "common/macros.h"
#include "core/framework/eplb/eplb_info.h"
#include "framework/model/causal_lm.h"
#include "platform/device.h"
#include "runtime/forward_params.h"

namespace xllm {

class EplbExecutor final {
 public:
  using Callback = std::function<void(int32_t)>;
  // The model outlives the executor: the worker owns the model and destroys
  // this executor before dropping the model. Passed by reference so the
  // executor cannot silently be constructed with a nullptr, and so the
  // ownership story is unambiguous (executor never claims the pointer).
  EplbExecutor(CausalLM& model, const torch::Device& device);

  virtual ~EplbExecutor();

  // Atomically return and clear the completed prepare-attempt token.
  int64_t consume_ready_prepare_token();

  // Execute EPLB operation based on coordination info
  // param eplb_info Contains layer preparation/activation instructions
  void eplb_execute(const EplbInfo& eplb_info);

  // Launch the pending layer's D2D transfer before model forward. Prepared
  // weights remain inactive until finish_eplb_step() is called.
  void start_eplb_step(const EplbInfo& eplb_info);

  // Wait for the transfer launched by start_eplb_step(), then atomically
  // activate the layer's weights and expert maps after model forward.
  void finish_eplb_step();

 private:
  struct Task {
    int32_t layer_id;
    int64_t prepare_token;
    std::vector<int32_t> expert_ids;
    Callback callback;
  };

  void eplb_worker_loop();
  void enqueue_prepare_task(Task task);
  void prepare_expert_weight_async(int32_t layer_id,
                                   int64_t prepare_token,
                                   const std::vector<int32_t>& expert_ids,
                                   Callback callback = nullptr);
  CausalLM& model_;
  Device device_;
  std::unique_ptr<Stream> stream_;
  std::queue<Task> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_ = false;

  mutable std::mutex ready_mutex_;
  int64_t ready_prepare_token_ = -1;

  std::mutex update_mutex_;
  int32_t pending_update_layer_id_ = -1;
  std::optional<Task> deferred_prepare_task_;
  // Declared last so every dependency used by eplb_worker_loop is fully
  // constructed before the thread starts and remains alive until it joins.
  std::thread eplb_worker_;
};

}  // namespace xllm

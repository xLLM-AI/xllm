/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "scheduler/chunked_prefill_scheduler.h"

namespace xllm {
class PrefillOnlyScheduler final : public ContinuousScheduler {
 public:
  PrefillOnlyScheduler(Engine* engine, const Options& options);
  virtual ~PrefillOnlyScheduler();

 private:
  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch() override;
  void handle_prefill_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      RequestPriorityQueue* waiting_priority_queue,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
  void handle_last_step_prefill_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& running_requests,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
};
}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <absl/time/time.h>

#include <memory>
#include <string>

#include "framework/request/request.h"

namespace xllm {

class SchedulerBase {
 public:
  virtual ~SchedulerBase() = default;

  // scheduler forward execute
  virtual void step(const absl::Duration& timeout) = 0;

  // offline running
  virtual void generate() = 0;

  // incr/decr pending requests
  virtual void incr_pending_requests(size_t count) {}
  virtual void decr_pending_requests() {}
  virtual size_t num_pending_requests() { return 0; }
};

class Scheduler : public SchedulerBase {
 public:
  virtual ~Scheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;

  virtual uint32_t get_waiting_requests_num() const = 0;

  virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                   std::vector<int64_t>& tbt) = 0;

  virtual const InstanceInfo& get_instance_info() = 0;

  // Drop all prefix-cache entries. Used by RL wake_up: discarded KV after deep
  // sleep makes cached prefixes point to garbage. Default no-op.
  virtual void reset_prefix_cache() {}
};

}  // namespace xllm

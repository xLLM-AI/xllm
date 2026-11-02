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

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/framework/config/execution_config.h"
#include "framework/request/request.h"

namespace xllm {

class CancelRequestQueue final {
 public:
  void submit(std::shared_ptr<Request> request) {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.emplace_back(std::move(request));
  }

  std::vector<std::shared_ptr<Request>> take_all() {
    std::vector<std::shared_ptr<Request>> requests;
    std::lock_guard<std::mutex> lock(mutex_);
    requests.swap(requests_);
    return requests;
  }

 private:
  std::mutex mutex_;
  std::vector<std::shared_ptr<Request>> requests_;
};

struct SchedulerOptions {
  // the maximum number of tokens per batch
  PROPERTY(int32_t, max_tokens_per_batch) = 20000;

  // the maximum number of sequences per batch
  PROPERTY(int32_t, max_seqs_per_batch) = 256;
  PROPERTY(bool, enable_task_pipeline) = false;

  // the capacity of the request queue; requests arriving while it is full
  // are rejected at admission.
  PROPERTY(int32_t, request_queue_size) = 100000;

  // the max tokens per chunk for request in prefill stage.
  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill);

  // the number of speculative tokens per step
  PROPERTY(int32_t, num_speculative_tokens) = 0;

  // Effective decode graph geometry for profile warmup. This is a resolved
  // scheduler value, not an Engine capability, because speculative target
  // width is known before the scheduler is constructed.
  PROPERTY(DecodeGraphWarmupConfig, decode_graph_warmup_config) = {};

  // the number of tp*dp*cp nodes
  PROPERTY(int32_t, nnodes) = 1;

  // the number of speculative tokens per step
  PROPERTY(int32_t, dp_size) = 1;

  PROPERTY(int32_t, cp_size) = 1;

  // enable disaggregated PD mode.
  PROPERTY(bool, enable_disagg_pd) = false;

  // for master service, current instance name(ID).
  PROPERTY(std::optional<std::string>, instance_name);

  PROPERTY(std::optional<InstanceRole>, instance_role) = InstanceRole::DEFAULT;

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  // In general decode instance send a batch responses to prefill in disagg pd
  // mode. here, we add a flag to control whether send a batch or single
  // response once, This will help us to debug code. default value is false.
  PROPERTY(bool, enable_batch_response) = false;

  // support P send batch reqs to D.
  // max_reqs_p2d_once represents the maximum number
  // of requests that can be sent once.
  // default value is 1.
  PROPERTY(int32_t, max_reqs_p2d_once) = 1;

  PROPERTY(bool, enable_schedule_overlap) = true;

  PROPERTY(bool, enable_chunked_prefill) = true;

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(bool, disable_log_stats) = false;

  // TODO: think if distinguish prefill and decode priority strategy
  PROPERTY(std::string,
           priority_strategy) = "fcfs";  // priority, deadline, fcfs

  PROPERTY(bool, enable_profile_step_time) = false;
  // use predicted latency for latency aware schedule
  PROPERTY(bool, enable_profile_token_budget) = false;

  PROPERTY(bool, enable_latency_aware_schedule) = false;
  // the max prompt length for profile
  PROPERTY(int32_t, profile_max_prompt_length) = 2048;
  // true if generate kv cache for profile
  PROPERTY(bool, enable_profile_kv_blocks) = true;
  // true if disable ttft profiling
  PROPERTY(bool, disable_ttft_profiling) = false;
  // all requests use single global ttft
  PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();
  // all requests use single global tpot
  PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

  // Index ID for internal server ID, which must be set different values
  // if the model supports multiple version or there are multiple models.
  PROPERTY(int64_t, server_idx) = 0;

  // max concurrency for rec worker
  PROPERTY(int32_t, rec_worker_max_concurrency) = 1;
};

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

  // Shutdown must keep advancing callbacks that still own scheduler state.
  virtual bool has_pending_prefetch() const { return false; }

  virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                   std::vector<int64_t>& tbt) = 0;

  virtual const InstanceInfo& get_instance_info() = 0;
};

}  // namespace xllm

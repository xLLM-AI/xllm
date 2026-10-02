/* Copyright 2025-2026 The xLLM Authors.

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
#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>

#include <atomic>
#include <concepts>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/batch_factory.h"
#include "core/framework/batch/batch_group.h"
#include "core/framework/block/kv_cache_manager.h"
#include "core/framework/request/priority_comparator.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/runtime/xservice_client.h"
#include "core/scheduler/async_response_processor.h"
#include "core/scheduler/profile/profile_manager.h"
#include "core/scheduler/request_priority_queue.h"
#include "core/scheduler/scheduler.h"
#include "core/scheduler/scheduler_metrics.h"

namespace xllm {
class RequestPriorityQueue;
class SchedulerConfig;
class SchedulerPolicy;
struct SchedulerState;

struct DecodeRestoreEntry {
  std::shared_ptr<Request> request;
  absl::Time started_at;
};

// BatchMode captures the scheduling policy configuration.
// The concrete SchedulerPolicy subclass is selected based on these fields:
//   - enable_mix_batch=false → PrefillFirstPolicy (exclusive batch)
//   - enable_mix_batch=true + priority_strategy!="multi_slo_and_prio" →
//   DecodeFirstPolicy
//   - enable_mix_batch=true + priority_strategy=="multi_slo_and_prio" →
//   UnifiedPolicy
//
// Mapping from old scheduler classes:
//   {false, false, "fcfs"}              → PrefillFirstPolicy (original
//   ContinuousScheduler) {true,  true,  "fcfs"}              →
//   DecodeFirstPolicy  (original ChunkedPrefillScheduler) {false, true, "fcfs"}
//   → PrefillFirstPolicy (original PrefillOnlyScheduler) {true,  true,
//   "multi_slo_and_prio"}   → UnifiedPolicy      (original MixScheduler)
struct BatchMode {
  bool enable_mix_batch = false;
  bool enable_chunked_prefill = false;
  // "fcfs": first-come-first-served
  // "multi_slo_and_prio": multi-priority multi-SLO aware scheduling (ProSched)
  // "priority": static priority weight
  // "deadline": earliest-deadline-first
  std::string priority_strategy = "fcfs";
};

class ContinuousSchedulerBase : public Scheduler {
 public:
  using Options = SchedulerOptions;
  using StepCallback = std::function<ForwardOutput(BatchGroup&)>;
  using ResultCallback = std::function<void(BatchGroup&)>;

  ~ContinuousSchedulerBase() override;

  bool add_request(std::shared_ptr<Request>& request) override;

  void step(const absl::Duration& timeout) override;

  void generate() override;

  // inc/dec pending requests
  void incr_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }
  void decr_pending_requests() override {
    const auto old_value =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(old_value, 0) << "pending requests underflow";
  }

  size_t num_pending_requests() override {
    return pending_requests_.load(std::memory_order_relaxed);
  }

  bool has_pending_prefetch() const override {
    return prefetching_requests_.load(std::memory_order_relaxed) > 0;
  }

  uint32_t get_waiting_requests_num() const override {
    return prefill_queue_->size() + chunk_queue_->size() +
           decode_restore_waiting_.size() +
           prefetching_requests_.load(std::memory_order_relaxed);
  }

  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt) override {}

  const InstanceInfo& get_instance_info() override { return instance_info_; }

 protected:
  ContinuousSchedulerBase(Engine* engine,
                          const Options& options,
                          StepCallback step_callback,
                          ResultCallback result_callback);

  void clear_mtp_bootstrap(Request* request);
  void drain_prefetch_pipeline();
  virtual void enqueue_ready_request(std::shared_ptr<Request> request);
  virtual void release_failed_request(const std::shared_ptr<Request>& request) {
  }

  // process the batch output
  void process_batch_output(bool enable_schedule_overlap);

  const Options options_;

  // BatchMode resolved from options/global config (subsumes old scheduler
  // hierarchy selection).
  BatchMode batch_mode_;

  // Process-wide scheduler configuration, resolved once at construction.
  const SchedulerConfig& scheduler_config_;

  // Each scheduler owns a factory configured for its DP topology.
  BatchFactory batch_factory_;

  // Policy object that encapsulates all batch-assembly logic.
  std::unique_ptr<SchedulerPolicy> policy_;

  // the engine to run the batch
  Engine* resource_engine_;

  StepCallback step_callback_;
  ResultCallback result_callback_;

  KVCacheManager* kv_cache_manager_;

  // a thread safe queue of requests, bounded by options_.request_queue_size()
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue_;

  std::atomic<size_t> prefetching_requests_{0};
  std::mutex prefetch_admission_mutex_;
  std::deque<std::shared_ptr<Request>> prefetch_admissions_;
  std::deque<std::shared_ptr<Request>> completed_prefetches_;

  // a batch of requests in running state, sorted by priority from high to low.
  // This may include decoding requests and prefill requests in chunked prefill
  // scheudler.
  std::vector<std::shared_ptr<Request>> running_requests_;

  // a batch of sequences that scheduled to run, sorted by priority from high to
  std::vector<Sequence*> running_sequences_;

  // token budget for each running sequence
  std::vector<size_t> running_sequences_budgets_;

  // preemptible requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<std::shared_ptr<Request>> preemptable_requests_;

  std::shared_ptr<CancelRequestQueue> cancel_request_queue_;

  std::unique_ptr<AsyncResponseProcessor> response_processor_;

  std::unique_ptr<ProfileManager> profile_manager_;
  std::unique_ptr<SchedulerMetrics> scheduler_metrics_;

  bool enable_prefix_cache_ = false;
  bool has_linear_attention_layers_ = false;
  bool enable_in_batch_prefix_cache_ = false;

  // the number of requests that are waiting to be scheduled
  std::atomic<size_t> pending_requests_{0};

  // Prefill queue: holds new prefill requests (kv_cache_tokens_num == 0).
  std::unique_ptr<RequestPriorityQueue> prefill_queue_;

  // Chunk queue: chunked prefill continuations (has partial KV cache,
  // can be preempted to free blocks when decode needs memory).
  std::unique_ptr<RequestPriorityQueue> chunk_queue_;

  // is last step handle prefill requests
  bool last_step_prefill_ = false;

  // Decode queue: holds all decode-stage requests.
  std::unique_ptr<RequestPriorityQueue> decode_queue_;

  // Decode victims that wait for D2H publication and device KV capacity before
  // re-entering the existing Prefill/H2D restore path.
  std::deque<DecodeRestoreEntry> decode_restore_waiting_;

  // Unified queue: used by UnifiedPolicy only (all requests in one queue).
  std::list<std::shared_ptr<Request>> unified_queue_;

  InstanceInfo instance_info_;

  int32_t min_speculative_tokens_required_ = 0;

  // build a batch of requests from the priority queue
  virtual BatchGroup prepare_batch();

  virtual bool if_queue_not_empty() {
    return !prefill_queue_->empty() || !chunk_queue_->empty() ||
           !decode_queue_->empty() || !decode_restore_waiting_.empty() ||
           !unified_queue_.empty();
  }

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  XServiceClient* xservice_client_ = nullptr;

  // params for enable_schedule_overlap case
  BatchGroup last_batch_;
  std::vector<std::shared_ptr<Request>> last_running_requests_;
  std::vector<Sequence*> last_running_sequences_;
  bool is_first_step_ = true;

  // Construct a SchedulerState snapshot for the policy.
  SchedulerState make_state();

  void drain_prefetch_admissions();
  void drain_completed_prefetches();

  void apply_cancel_requests();

  void drain_decode_restore_waiting(
      std::vector<std::shared_ptr<Request>>& finished);

  BatchGroup schedule_request(const absl::Duration& timeout);

  void step_with_schedule_overlap(const absl::Duration& timeout);

  void refresh_sequences_from_requests(
      const std::vector<std::shared_ptr<Request>>& requests,
      std::vector<Sequence*>& sequences) const;

  void create_queues(const Options& options);
};

template <typename EngineType = Engine>
  requires std::derived_from<EngineType, Engine>
class ContinuousScheduler : public ContinuousSchedulerBase {
 public:
  using Options = SchedulerOptions;

  ContinuousScheduler(EngineType* engine, const Options& options)
    requires requires(EngineType* typed_engine, BatchGroup& batch) {
      { typed_engine->step(batch) } -> std::same_as<ForwardOutput>;
      { typed_engine->update_last_step_result(batch) } -> std::same_as<void>;
    }
      : ContinuousSchedulerBase(
            engine,
            options,
            [engine](BatchGroup& batch) { return engine->step(batch); },
            [engine](BatchGroup& batch) {
              engine->update_last_step_result(batch);
            }),
        engine_(engine) {}

  template <typename TargetEngine>
    requires std::same_as<EngineType, Engine> &&
                 requires(TargetEngine* typed_engine, BatchGroup& batch) {
                   static_cast<Engine*>(typed_engine);
                   { typed_engine->step(batch) } -> std::same_as<ForwardOutput>;
                   {
                     typed_engine->update_last_step_result(batch)
                   } -> std::same_as<void>;
                 }
  ContinuousScheduler(TargetEngine* engine, const Options& options)
      : ContinuousSchedulerBase(
            static_cast<Engine*>(engine),
            options,
            [engine](BatchGroup& batch) { return engine->step(batch); },
            [engine](BatchGroup& batch) {
              engine->update_last_step_result(batch);
            }),
        engine_(static_cast<Engine*>(engine)) {}

  ~ContinuousScheduler() override = default;

 protected:
  EngineType* engine_;
};

}  // namespace xllm

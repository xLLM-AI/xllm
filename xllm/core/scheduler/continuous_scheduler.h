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

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>

#include <atomic>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "async_response_processor.h"
#include "common/macros.h"
#include "common/types.h"
#include "core/framework/config/rec_config.h"
#include "framework/batch/batch.h"
#include "framework/block/kv_cache_manager.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/xservice_client.h"
#include "scheduler.h"
#include "scheduler/profile/profile_manager.h"
#include "scheduler/request_priority_queue.h"

namespace xllm {
class Engine;
class RequestPriorityQueue;

class CancelRequestQueue final {
 public:
  void submit(std::shared_ptr<Request> request);
  std::vector<std::shared_ptr<Request>> take_all();

 private:
  std::mutex mutex_;
  std::vector<std::shared_ptr<Request>> requests_;
};

class ContinuousScheduler : public Scheduler {
 public:
  struct Options {
    // the maximum number of tokens per batch
    PROPERTY(int32_t, max_tokens_per_batch) = 20000;

    // the maximum number of sequences per batch
    PROPERTY(int32_t, max_seqs_per_batch) = 256;

    // the max tokens per chunk for request in prefill stage.
    PROPERTY(int32_t, max_tokens_per_chunk_for_prefill);

    // the number of speculative tokens per step
    PROPERTY(int32_t, num_speculative_tokens) = 0;

    // the number of tp*dp*cp nodes
    PROPERTY(int32_t, nnodes) = 1;

    // the number of speculative tokens per step
    PROPERTY(int32_t, dp_size) = 1;

    PROPERTY(int32_t, cp_size) = 1;

    // enable disaggregated PD mode.
    PROPERTY(bool, enable_disagg_pd) = false;

    PROPERTY(bool, enable_pd_ooc) = false;

    // for master service, current instance name(ID).
    PROPERTY(std::optional<std::string>, instance_name);

    PROPERTY(std::optional<InstanceRole>,
             instance_role) = InstanceRole::DEFAULT;

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
    PROPERTY(bool, enable_online_preempt_offline) = true;

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
    // true if enable forward interruption
    PROPERTY(bool, enable_forward_interruption) = false;
    // all requests use single global ttft
    PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();
    // all requests use single global tpot
    PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

    // Index ID for internal server ID, which must be set different values
    // if the model supports multiple version or there are multiple models.
    PROPERTY(int64_t, server_idx) = 0;

    // Prefetch timeout for prefetch from kv cache store
    PROPERTY(uint32_t, prefetch_timeout) = 0;

    // max concurrency for rec worker
    PROPERTY(int32_t, rec_worker_max_concurrency) = 1;
  };

  ContinuousScheduler(Engine* engine, const Options& options);
  virtual ~ContinuousScheduler();

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

  size_t num_pending_requests() {
    return pending_requests_.load(std::memory_order_relaxed);
  }

  uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_->size() +
           waiting_priority_queue_offline_->size();
  }

  // for test only
  std::vector<Batch> prepare_batch_test() { return prepare_batch(); }
  std::vector<std::shared_ptr<Request>> get_running_requests() {
    return running_requests_;
  }
  std::vector<size_t> get_running_sequences_budgets() {
    return running_sequences_budgets_;
  }
  std::vector<std::shared_ptr<Request>> get_waiting_requests() {
    std::vector<std::shared_ptr<Request>> result;
    if (waiting_priority_queue_ == nullptr) {
      return result;
    }

    auto copied_waiting_queue = waiting_priority_queue_->clone();
    result.reserve(copied_waiting_queue->size());
    while (!copied_waiting_queue->empty()) {
      result.emplace_back(copied_waiting_queue->top());
      copied_waiting_queue->pop_top();
    }

    return result;
  }

  ProfileManager* get_profile_manager() { return profile_manager_.get(); }

  virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                   std::vector<int64_t>& tbt) {}

  const InstanceInfo& get_instance_info() { return instance_info_; }

  std::vector<int> last_batch_lengths_;

  // Async RL training support: pause/resume
  enum class PauseState {
    RUNNING = 0,  // Normal operation
    PAUSING = 1,  // Requested pause, transitioning to PAUSED
    PAUSED = 2    // Fully paused
  };

  // How to handle in-flight requests when pausing (vLLM-compatible).
  enum class PauseMode {
    KEEP = 0,   // Preempt running requests, free KV cache, push back to waiting
                // queue; recomputed (re-prefill) on resume. Default for RL.
    ABORT = 1,  // Cancel all running requests; clients must retry.
    WAIT = 2    // Stop admitting new requests; let running requests finish
                // naturally, then pause. KV cache is not discarded.
  };

  // Pause the scheduler. See PauseMode for in-flight request handling.
  void reset_prefix_cache() override;

  void pause(PauseMode mode = PauseMode::KEEP);

  // Block until the scheduler has fully transitioned to PAUSED (i.e. running
  // requests have been handled per mode and it is safe to update weights).
  // Returns true if paused, false if it timed out first.
  bool wait_until_paused(int64_t timeout_ms = -1);

  // Resume the scheduler.
  void resume();

  // Check if scheduler is paused or pausing
  bool is_paused() const;

  // for test only: directly trigger the preemption that step() performs when
  // transitioning to PAUSED, without needing a real engine to drive step().
  void preempt_all_running_requests_test() { preempt_all_running_requests(); }
  void abort_all_running_requests_test() { abort_all_running_requests(); }

 private:
  // Drive the PAUSING -> PAUSED transition from within step(). Returns true if
  // the scheduler is paused (caller should skip normal scheduling).
  bool try_complete_pause();

  // KEEP mode: preempt all running requests, free KV cache, push to waiting.
  void preempt_all_running_requests();

  // ABORT mode: cancel all running requests; they are not rescheduled.
  void abort_all_running_requests();

  void clear_mtp_bootstrap(Request* request);

 protected:
  // Rounds a per-step wall-clock latency to an amortized per-token latency,
  // i.e. round(tbt_ms / num_tokens). num_tokens must be > 0.
  static int64_t amortized_token_latency_ms(int64_t tbt_ms, size_t num_tokens);

  const Options options_;

  // the engine to run the batch
  Engine* engine_;

  KVCacheManager* kv_cache_manager_;

  // a thread safe queue of requests, bounded by
  // ::xllm::RecConfig::get_instance().request_queue_size() the schedule
  // owns the requests and manages their lifetimes.
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue_;

  // a batch of requests in running state, sorted by priority from high to low.
  // This may include decoding requests and prefill requests in chunked prefill
  // scheudler.
  std::vector<std::shared_ptr<Request>> running_requests_;

  // a batch of sequences that scheduled to run, sorted by priority from high to
  std::vector<Sequence*> running_sequences_;

  // token budget for each running sequence
  std::vector<size_t> running_sequences_budgets_;

  // preemptable requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<std::shared_ptr<Request>> preemptable_requests_;

  std::shared_ptr<CancelRequestQueue> cancel_request_queue_;

  std::unique_ptr<AsyncResponseProcessor> response_processor_;

  std::unique_ptr<ProfileManager> profile_manager_;

  bool enable_prefix_cache_ = false;

  // Cached once at construction (mirrors enable_prefix_cache_): the model's
  // layer set is immutable, so the hot scheduling path must not re-scan
  // layer_types() with std::any_of on every allocate.
  bool has_linear_attention_layers_ = false;

  bool enable_in_batch_prefix_cache_ = false;

  // the number of requests that are waiting to be scheduled
  std::atomic<size_t> pending_requests_{0};

  // keep all new requests, generally speaking, they do not have any kv cache.
  std::unique_ptr<RequestPriorityQueue> waiting_priority_queue_;
  std::unique_ptr<RequestPriorityQueue> waiting_priority_queue_offline_;

  // keep all running request from high priority to low.
  // NOTE: Maybe not all requests are scheduled in one step,
  // this is decided by the kv blocks usage.
  // This contain all decoding requests and requests that have been
  // popped from waiting_priority_queue_ but remain in prefill stage,
  // these requests have already allocated some kv caches,
  // so they can be preemeted in scheduler.

  // is last step handle prefill requests
  bool last_step_prefill_ = false;

  // std::deque<std::shared_ptr<Request>> running_queue_;
  // std::deque<std::shared_ptr<Request>> running_queue_offline_;
  std::unique_ptr<RequestPriorityQueue> running_queue_;
  std::unique_ptr<RequestPriorityQueue> running_queue_offline_;

  InstanceInfo instance_info_;

  int32_t min_speculative_tokens_required_ = 0;

  virtual void handle_prefill_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      RequestPriorityQueue* waiting_priority_queue,
      size_t& num_online_prefill_preempt_offline_requests,
      std::vector<std::shared_ptr<Request>>& finished_requests);
  virtual void handle_decode_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_offline_decode_preempt_offline_requests,
      size_t& num_online_decode_preempt_online_requests,
      size_t& num_online_decode_preempt_offline_requests,
      RequestPriorityQueue* running_queue);
  void get_latency_budget_and_request_order(
      RequestPriorityQueue* request_priority_queue,
      double& latency_budget,
      bool for_prefill);

  void handle_abnormal_request(
      RequestPriorityQueue* running_queue,
      const std::vector<Sequence*>& candidate_sequences,
      const std::vector<size_t>& candidate_token_budgets,
      const size_t& allocated_tokens,
      const size_t& allocated_seqs,
      double& allocated_estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      double& estimate_latency,
      bool budget_exhausted,
      bool block_exhausted);
  void handle_running_requests(std::shared_ptr<Request> request);

  bool check_if_enough_to_evict(RequestPriorityQueue* running_queue_to_evict,
                                Sequence* prefill_sequence,
                                size_t max_handle_num_tokens,
                                size_t& num_request_to_evict);

  // Publish the full prefill blocks admitted in this scheduling step into the
  // prefix cache, so later requests in the same batch can share them.
  void cache_in_batch_prefix(
      const std::vector<Sequence*>& sequences,
      const std::vector<size_t>& current_step_token_budgets);

  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch();

  virtual bool if_queue_not_empty() {
    return !waiting_priority_queue_->empty() || !running_queue_->empty() ||
           !waiting_priority_queue_offline_->empty() ||
           !running_queue_offline_->empty();
  }

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  XServiceClient* xservice_client_ = nullptr;

  // params for enable_schedule_overlap case
  std::vector<Batch> last_batch_;
  std::vector<std::shared_ptr<Request>> last_running_requests_;
  std::vector<Sequence*> last_running_sequences_;
  bool is_first_step_ = true;

  // Pause state (atomic for thread-safe access)
  std::atomic<PauseState> pause_state_{PauseState::RUNNING};
  // How to handle in-flight requests for the current pause. Only read while
  // pause_state_ != RUNNING; written by pause() before publishing the state.
  std::atomic<PauseMode> pause_mode_{PauseMode::KEEP};
  // Signals the PAUSING -> PAUSED transition to callers blocked in
  // wait_until_paused(). Notified by the scheduler loop thread.
  std::mutex pause_mutex_;
  std::condition_variable pause_cv_;

 private:
  void apply_cancel_requests();

  std::vector<Batch> schedule_request(const absl::Duration& timeout);

  virtual void update_token_latency_metrics(std::vector<Sequence*>& sequences);

  // process the batch output
  void process_batch_output(bool enable_schedule_overlap);

  void step_with_schedule_overlap(const absl::Duration& timeout);

  void step_with_pd_ooc(std::vector<Batch>& batch);

  void refresh_sequences_from_requests(
      const std::vector<std::shared_ptr<Request>>& requests,
      std::vector<Sequence*>& sequences) const;

  std::vector<int64_t> get_num_occupied_slots(
      std::vector<Sequence*>& sequences) const;
  std::vector<int64_t> get_active_activation_in_bytes();
  void update_memory_metrics(std::vector<Sequence*>& sequences);

  void create_waiting_queue(const Options& options);
  void create_running_queue(const Options& options);
};

}  // namespace xllm

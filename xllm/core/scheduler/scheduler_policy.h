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

#include <folly/MPMCQueue.h>

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

#include "framework/batch/batch.h"
#include "framework/block/kv_cache_manager.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/profile/profile_manager.h"
#include "scheduler/request_priority_queue.h"

namespace xllm {

class AsyncResponseProcessor;

// SchedulerState provides explicit access to the scheduler's internal state.
// The SchedulerPolicy operates solely through this boundary -- it never
// accesses ContinuousScheduler members directly.
struct SchedulerState {
  // Queues (unified -- no separate online/offline distinction).
  // prefill_queue: holds new prefill requests (kv_cache_tokens_num == 0)
  // chunk_queue: holds chunked prefill continuations (has partial KV,
  // preemptable) decode_queue: holds decode requests unified_queue: used by
  // UnifiedPolicy (all requests in one list)
  RequestPriorityQueue& prefill_queue;
  RequestPriorityQueue& chunk_queue;
  RequestPriorityQueue& decode_queue;
  std::list<std::shared_ptr<Request>>& unified_queue;

  // Current batch state (reset each step).
  std::vector<std::shared_ptr<Request>>& running_requests;
  std::vector<Sequence*>& running_sequences;
  std::vector<size_t>& running_sequences_budgets;

  // Infrastructure.
  KVCacheManager* kv_cache_manager;
  ProfileManager* profile_manager;
  AsyncResponseProcessor* response_processor;

  // Flags.
  bool& last_step_prefill;

  // Configuration.
  const ContinuousScheduler::Options& options;
  int32_t min_speculative_tokens_required;
  bool enable_prefix_cache;
  bool has_linear_attention_layers;
};

// ScheduleBudget tracks the remaining resources for the current scheduling
// step. Methods consume from these budgets as they add sequences to the batch.
struct ScheduleBudget {
  size_t remaining_token_budget;
  size_t remaining_seq_budget;
  double latency_budget;
  double estimate_latency;
  size_t num_preempted_requests;
};

inline bool budget_exhausted(const ScheduleBudget& budget) {
  return budget.remaining_token_budget == 0 ||
         budget.remaining_seq_budget == 0 ||
         budget.latency_budget <= budget.estimate_latency;
}

// Selects which concrete policy to instantiate.
// Determined by BatchMode: enable_mix_batch × priority_strategy.
enum class SchedulerPolicyKind {
  PREFILL_FIRST,  // !enable_mix_batch: prefill and decode never share a batch
  DECODE_FIRST,   // enable_mix_batch + fcfs/priority/deadline: decode fills
                  // first
  UNIFIED,  // enable_mix_batch + multi_slo_and_prio: unified queue scheduling
};

// SchedulerPolicy is the abstract base for batch-assembly strategies.
//
// ContinuousScheduler owns a SchedulerPolicy and calls prepare_batch() each
// step. The policy handles: requeue previous running requests → schedule new
// batch. Which concrete policy is instantiated depends on BatchMode (see
// continuous_scheduler.h).
//
// Common scheduling primitives (schedule_prefill_from_queue,
// schedule_decode_from_queue, token computation, block allocation, preemption)
// live in the base class. Each subclass only implements the top-level
// scheduling order and queue management strategy.
class SchedulerPolicy {
 public:
  SchedulerPolicy(const BatchMode& mode,
                  const ContinuousScheduler::Options& options);
  virtual ~SchedulerPolicy() = default;

  // The strategy-specific scheduling logic.
  // Called by ContinuousScheduler::prepare_batch() after drain + budget init.
  virtual void schedule(SchedulerState& state,
                        ScheduleBudget& budget,
                        std::vector<std::shared_ptr<Request>>& finished) = 0;

  // Common phases called by ContinuousScheduler::prepare_batch().
  virtual void drain_request_queue(
      SchedulerState& state,
      folly::MPMCQueue<std::shared_ptr<Request>>& request_queue);
  std::vector<std::shared_ptr<Request>> collect_finished(SchedulerState& state);
  void report_metrics(const SchedulerState& state,
                      double elapsed_seconds,
                      size_t num_preempted_requests);

 protected:
  // ===== Common phases (internal) =====
  void reset_batch_state(SchedulerState& state);

  // ===== Running request state update =====
  void handle_running_requests(std::shared_ptr<Request> request,
                               SchedulerState& state);

  // Adjusts latency budget and reorders queues based on SLO urgency.
  // Default: no-op. Subclasses override for multi_slo_and_prio scheduling.
  // second_queue can be nullptr if only one queue needs processing.
  virtual void adjust_latency_budget_and_reorder(
      RequestPriorityQueue* first_queue,
      RequestPriorityQueue* second_queue,
      double& latency_budget,
      bool for_prefill,
      const SchedulerState& state);

  // ===== Prefill scheduling =====
  void schedule_prefill_from_queue(
      RequestPriorityQueue* queue,
      SchedulerState& state,
      ScheduleBudget& budget,
      std::vector<std::shared_ptr<Request>>& finished);
  size_t compute_prefill_tokens(Sequence* seq,
                                size_t remaining_budget,
                                const SchedulerState& state);
  bool allocate_for_prefill(Sequence* seq,
                            size_t token_budget,
                            size_t* actual_tokens,
                            SchedulerState& state,
                            bool skip_shared = false);
  void allocate_shared_blocks_for(Sequence* seq, SchedulerState& state);

  // ===== Decode scheduling =====
  void schedule_decode_from_queue(RequestPriorityQueue* queue,
                                  SchedulerState& state,
                                  ScheduleBudget& budget);

  // ===== Preemption and error handling =====
  void handle_unschedulable_head(
      RequestPriorityQueue* queue,
      SchedulerState& state,
      std::vector<std::shared_ptr<Request>>& finished,
      bool budget_exhausted,
      bool blocks_exhausted);
  void handle_abnormal_decode_request(
      RequestPriorityQueue* queue,
      SchedulerState& state,
      ScheduleBudget& budget,
      std::shared_ptr<Request>& request,
      const std::vector<Sequence*>& candidate_sequences,
      const std::vector<size_t>& candidate_token_budgets,
      size_t allocated_tokens,
      size_t allocated_seqs,
      double allocated_estimate_latency,
      bool budget_exhausted);

  // ===== Helpers =====
  void cache_in_batch_prefix(const std::vector<Sequence*>& sequences,
                             const std::vector<size_t>& token_budgets,
                             SchedulerState& state);
  void clear_mtp_bootstrap(Request* request, const SchedulerState& state);

  BatchMode batch_mode_;
  const ContinuousScheduler::Options& options_;
};

// =============================================================================
// Concrete policies
// =============================================================================

// PrefillFirstPolicy: exclusive batch mode (!enable_mix_batch).
// Each batch is either all-prefill or all-decode, never mixed.
// Prefill is prioritized; decode only runs when no prefill is available.
// Supports all priority_strategy values for per-queue ordering.
class PrefillFirstPolicy : public SchedulerPolicy {
 public:
  using SchedulerPolicy::SchedulerPolicy;

  void schedule(SchedulerState& state,
                ScheduleBudget& budget,
                std::vector<std::shared_ptr<Request>>& finished) override;

  void adjust_latency_budget_and_reorder(RequestPriorityQueue* first_queue,
                                         RequestPriorityQueue* second_queue,
                                         double& latency_budget,
                                         bool for_prefill,
                                         const SchedulerState& state) override;
};

// DecodeFirstPolicy: mixed batch mode with fcfs/priority/deadline.
// Decode requests fill the batch first ("decode-maximal batching"),
// remaining token budget goes to chunked prefill.
// Implements the Sarathi-Serve strategy (arxiv 2403.02310).
class DecodeFirstPolicy : public SchedulerPolicy {
 public:
  using SchedulerPolicy::SchedulerPolicy;

  void schedule(SchedulerState& state,
                ScheduleBudget& budget,
                std::vector<std::shared_ptr<Request>>& finished) override;

 private:
  void redistribute_remaining_budget(SchedulerState& state,
                                     ScheduleBudget& budget,
                                     std::vector<Sequence*>& prefill_seqs);
};

// UnifiedPolicy: prefill and decode requests are merged into a single queue
// and sorted/scheduled together, regardless of their stage.
// Currently used for multi_slo_and_prio (multi-priority multi-SLO scheduling),
// where high-priority decode and prefill both rank above low-priority requests.
// Developers can inherit this class to implement custom unified scheduling
// strategies.
class UnifiedPolicy : public SchedulerPolicy {
 public:
  using SchedulerPolicy::SchedulerPolicy;

  void drain_request_queue(
      SchedulerState& state,
      folly::MPMCQueue<std::shared_ptr<Request>>& request_queue) override;

  void schedule(SchedulerState& state,
                ScheduleBudget& budget,
                std::vector<std::shared_ptr<Request>>& finished) override;

 private:
  void schedule_from_unified_queue(
      std::list<std::shared_ptr<Request>>& unified,
      SchedulerState& state,
      ScheduleBudget& budget,
      std::vector<std::shared_ptr<Request>>& finished);
  void get_latency_budget_and_request_order(
      std::list<std::shared_ptr<Request>>& queue,
      double& latency_budget,
      const SchedulerState& state);
  size_t get_max_copy_block_num(std::list<std::shared_ptr<Request>>& queue,
                                ScheduleBudget& budget,
                                const SchedulerState& state);
  size_t get_needed_copy_block_num(
      std::vector<std::shared_ptr<Request>>& req_vec,
      std::vector<size_t>& per_req_copy_block_num_vec,
      double max_h2d_transfer_time,
      double min_total_exec_time,
      size_t max_h2d_block_num,
      const SchedulerState& state);
  int32_t get_max_chunk(Sequence* sequence,
                        size_t num_tokens,
                        size_t kv_cache_tokens_num,
                        int32_t latency_budget,
                        const SchedulerState& state);
};

// Factory function: creates the appropriate policy based on BatchMode.
std::unique_ptr<SchedulerPolicy> create_scheduler_policy(
    const BatchMode& mode,
    const ContinuousScheduler::Options& options);

}  // namespace xllm

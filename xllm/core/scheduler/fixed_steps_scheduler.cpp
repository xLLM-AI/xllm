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

#include "core/scheduler/fixed_steps_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/framework/batch/rec_batch.h"
#include "core/framework/batch/rec_batch_factory.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/request/priority_comparator.h"
#include "core/framework/request/rec_type.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/util/rec_model_utils.h"
#include "core/util/timer.h"

namespace xllm {

FixedStepsScheduler::FixedStepsScheduler(RecEngine* engine,
                                         const Options& options)
    : options_(options),
      engine_(engine),
      request_queue_(options.request_queue_size()),
      notifier_(std::make_shared<SchedulerNotifier>()),
      step_semaphore_(
          static_cast<std::ptrdiff_t>(options.rec_worker_max_concurrency())) {
  CHECK(engine_ != nullptr);
  kv_cache_manager_ = engine_->block_manager_pool();
  CHECK(kv_cache_manager_ != nullptr);
  kv_cache_manager_->set_prefetch_completion_callback(
      [notifier = notifier_]() { notifier->notify(); });

  enable_prefix_cache_ =
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();

  cancel_request_queue_ = std::make_shared<CancelRequestQueue>();
  output_dispatcher_ = std::make_unique<RequestOutputDispatcher>(
      engine_->tokenizer(),
      options_.enable_service_routing(),
      options_.disable_log_stats(),
      [cancel_request_queue = cancel_request_queue_,
       notifier = notifier_](std::shared_ptr<Request> request) {
        cancel_request_queue->submit(std::move(request));
        notifier->notify();
      });

  if (options_.priority_strategy() == "multi_slo_and_prio" ||
      options_.priority_strategy() == "fcfs") {
    prefill_queue_ = std::make_unique<DequeQueue>();
  } else {
    auto comparator = create_comparator(options_.priority_strategy(),
                                        /*reverse=*/false);
    prefill_queue_ = std::make_unique<HeapQueue>(std::move(comparator));
  }

  instance_info_.name = options_.instance_name().value_or("");
  instance_info_.type = options_.instance_role().value().to_string();
  instance_info_.dp_size = options_.dp_size();
  instance_info_.kv_split_size =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();

  step_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/static_cast<size_t>(options.rec_worker_max_concurrency()),
      /*cpu_binding=*/false,
      /*pool_name=*/"FixedStepsScheduler.step");
}

FixedStepsScheduler::~FixedStepsScheduler() {
  notifier_->request_stop();
  kv_cache_manager_->set_prefetch_completion_callback({});

  // Tasks capture scheduler state and must finish before it is destroyed.
  step_threadpool_.reset();

  // Unissued admissions have no asynchronous callback and can be cancelled
  // directly, including an offline scheduler that never started.
  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    for (const std::shared_ptr<Request>& request : prefetch_admissions_) {
      request->set_cancel();
    }
    const size_t unissued = prefetch_admissions_.size();
    prefetch_admissions_.clear();
    const size_t previous =
        prefetching_requests_.fetch_sub(unissued, std::memory_order_acq_rel);
    CHECK_GE(previous, unissued);
  }
  kv_cache_manager_->drain_prefetch_completions();
  drain_completed_prefetches();
  CHECK_EQ(prefetching_requests_.load(std::memory_order_acquire), 0u)
      << "FixedStepsScheduler destroyed with pending prefetch callbacks";
  running_requests_.clear();
  output_dispatcher_.reset();
}

bool FixedStepsScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
  const size_t pending_before_reservation =
      prefetching_requests_.load(std::memory_order_relaxed);
  const size_t queued_requests =
      static_cast<size_t>(std::max<ssize_t>(request_queue_.size(), 0));
  if (queued_requests + pending_before_reservation >=
      request_queue_.capacity()) {
    return false;
  }

  if (!kv_cache_manager_->has_storage_prefetch()) {
    const bool accepted = request_queue_.write(request);
    if (accepted) {
      notifier_->notify();
    }
    return accepted;
  }

  prefetching_requests_.fetch_add(1, std::memory_order_relaxed);
  prefetch_admissions_.emplace_back(request);
  notifier_->notify();
  VLOG(1) << "[Mooncake][AdmissionPending] request=" << request->request_id();
  return true;
}

void FixedStepsScheduler::drain_prefetch_admissions() {
  std::deque<std::shared_ptr<Request>> requests;
  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    requests.swap(prefetch_admissions_);
  }

  for (std::shared_ptr<Request>& request : requests) {
    if (request->finished() || request->cancelled()) {
      const size_t previous =
          prefetching_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
      continue;
    }

    for (const auto& sequence : request->sequences()) {
      if (sequence->dp_rank() < 0) {
        sequence->set_dp_rank(
            dp_rank_selector_.select(kv_cache_manager_->dp_rank_loads()));
        CHECK_GE(sequence->dp_rank(), 0);
      }
    }
    kv_cache_manager_->prefetch_from_storage(
        request, [this](std::shared_ptr<Request> completed) {
          std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
          completed_prefetches_.emplace_back(std::move(completed));
        });
  }
}

void FixedStepsScheduler::drain_completed_prefetches() {
  std::deque<std::shared_ptr<Request>> completed;
  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    completed.swap(completed_prefetches_);
  }

  for (std::shared_ptr<Request>& request : completed) {
    const bool cancelled = request->finished() || request->cancelled();
    {
      // Admission observes the queue size and reservation count together.
      std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
      if (!cancelled) {
        CHECK(request_queue_.write(request))
            << "Reserved request queue slot disappeared before prefetch "
               "completed.";
      }
      const size_t previous =
          prefetching_requests_.fetch_sub(1, std::memory_order_relaxed);
      CHECK_GT(previous, 0u);
    }
    VLOG(1) << (cancelled ? "[Mooncake][AdmissionCancelled] request="
                          : "[Mooncake][AdmissionReady] request=")
            << request->request_id();
  }
}

void FixedStepsScheduler::drain_prefetch_pipeline() {
  kv_cache_manager_->drain_prefetch_completions();
  drain_prefetch_admissions();
  kv_cache_manager_->drain_prefetch_completions();
  drain_completed_prefetches();
}

void FixedStepsScheduler::apply_cancel_requests() {
  std::vector<std::shared_ptr<Request>> requests =
      cancel_request_queue_->take_all();
  for (const std::shared_ptr<Request>& request : requests) {
    request->set_cancel();
  }
}

bool FixedStepsScheduler::handle_prefill_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // Handle new request prompt first.
  // Include those requests that are preempted by others.
  //
  // schedule the prefill requests in the waiting priority queue until budgets
  // are exhausted.
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  //
  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contain many sequences, so we should check here.
  bool rejected_oversized_request = false;
  bool budget_exhausted = false;
  const bool requires_kv_cache =
      scheduler_pipeline_ && scheduler_pipeline_->requires_kv_cache();
  while (!prefill_queue_->empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 &&
         kv_cache_manager_->kv_cache_utilization() <
             ::xllm::SchedulerConfig::get_instance()
                 .prefill_scheduling_memory_usage_threshold()) {
    std::shared_ptr<Request> request(prefill_queue_->top());
    if (request->finished() || request->cancelled()) {
      if (requires_kv_cache) {
        kv_cache_manager_->deallocate(request.get());
      }
      //  release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      prefill_queue_->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request has an unexpected sequence count.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    bool can_schedule = true;
    bool exceeds_total_kv_capacity = false;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    std::vector<int32_t> allocation_dp_ranks;
    std::vector<KVCacheManager::KVCacheReservation> prefill_reservations;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    if (requires_kv_cache) {
      prefill_reservations.reserve(request->sequences().size());
    }
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      if (!requires_kv_cache && prefill_sequence->dp_rank() < 0) {
        prefill_sequence->set_dp_rank(0);
      }

      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      if (requires_kv_cache) {
        const auto reservation =
            scheduler_pipeline_->kv_cache_reservation(prefill_sequence.get());
        if (!reservation.has_value()) {
          can_schedule = false;
          exceeds_total_kv_capacity = true;
          break;
        }
        prefill_reservations.emplace_back(*reservation);
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
    }

    if (can_schedule && requires_kv_cache &&
        kv_cache_manager_->exceeds_total_capacity(prefill_reservations)) {
      can_schedule = false;
      exceeds_total_kv_capacity = true;
    }

    size_t num_allocated_sequences = 0;
    if (can_schedule && requires_kv_cache) {
      allocation_dp_ranks.reserve(prefill_reservations.size());
      // Total capacity was checked for the complete request above. Allocate
      // the same reservations; any failure here is temporary cache pressure.
      for (const auto& [sequence, num_tokens] : prefill_reservations) {
        allocation_dp_ranks.emplace_back(sequence->dp_rank());
        if (sequence->dp_rank() < 0) {
          sequence->set_dp_rank(
              dp_rank_selector_.select(kv_cache_manager_->dp_rank_loads()));
          CHECK_GE(sequence->dp_rank(), 0);
        }
        if (!kv_cache_manager_->allocate(sequence, num_tokens)) {
          can_schedule = false;
          break;
        }
        ++num_allocated_sequences;
      }
    }

    if (!can_schedule) {
      for (size_t i = 0; i < num_allocated_sequences; ++i) {
        kv_cache_manager_->deallocate(prefill_sequences[i]);
      }
      // Failed allocation may also select a rank. Restore every attempted
      // sequence so temporary pressure cannot pin the request to one pool.
      for (size_t i = 0; i < allocation_dp_ranks.size(); ++i) {
        prefill_sequences[i]->set_dp_rank(allocation_dp_ranks[i]);
      }
      if (exceeds_total_kv_capacity) {
        prefill_queue_->pop_top();
        output_dispatcher_->process_failed_request(
            request,
            {StatusCode::RESOURCE_EXHAUSTED,
             "KV cache request exceeds total pool capacity."});
        rejected_oversized_request = true;
        continue;
      }
      break;
    }

    if (prefill_sequences.empty()) {
      continue;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    prefill_queue_->pop_top();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
  }

  if (running_sequences_.empty() && !prefill_queue_->empty() &&
      budget_exhausted) {
    LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                  "a single sequence.";
    std::shared_ptr<Request> request(prefill_queue_->top());
    prefill_queue_->pop_top();
    // block_manager_->release_blocks_for(request.get());
    output_dispatcher_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough budget to schedule single sequence."});
    rejected_oversized_request = true;
  }
  return rejected_oversized_request && !prefill_queue_->empty();
}

RecBatchGroup FixedStepsScheduler::prepare_rec_batch() {
  Timer timer;
  apply_cancel_requests();
  drain_prefetch_pipeline();
  // propagate new requests to prefill_queue_
  // Include those requests that are preempted by others.
  auto propagate_request = [this](std::shared_ptr<Request>& request) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      prefill_queue_->push(request);
    } else {
      // request from prefill instance in disagge pd mode.
      running_requests_.emplace_back(request);
    }
  };

  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
  while (request_queue_.read(request)) {
    propagate_request(request);
  }

  // Select the request type before touching running requests. Disaggregated
  // PD requests skip the prefill queue and may be the first request seen by
  // this scheduler.
  if (!scheduler_pipeline_) {
    std::shared_ptr<Request> sample_request;
    if (!prefill_queue_->empty()) {
      sample_request = prefill_queue_->top();
    } else {
      for (const auto& candidate : running_requests_) {
        if (candidate != nullptr) {
          sample_request = candidate;
          break;
        }
      }
    }
    if (sample_request != nullptr) {
      const auto rec_type = sample_request->state().rec_type;
      const bool is_rec_multi_round =
          (rec_type == RecType::kLlmRec) && is_rec_multi_round_mode();
      scheduler_pipeline_ =
          create_scheduler_pipeline(rec_type, is_rec_multi_round);
      rec_batch_factory_ = std::make_unique<RecBatchFactory>(
          options_.dp_size(), scheduler_pipeline_->input_type());
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      if (scheduler_pipeline_->requires_kv_cache()) {
        kv_cache_manager_->deallocate(request.get());
      }
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;

  while (handle_prefill_requests(
      remaining_token_budget, remaining_seq_budget, finished_requests)) {
    // Continue immediately after rejecting an oversized queue head.
  }

  // only forward once, no decode requests
  // handle_decode_requests(
  //     remaining_token_budget, remaining_seq_budget, num_preempted_requests);

  if (!finished_requests.empty()) {
    output_dispatcher_->process_completed_requests(finished_requests);
  }

  RecBatchGroup batches;
  if (rec_batch_factory_) {
    batches = rec_batch_factory_->create_batches(
        running_requests_,
        running_sequences_,
        running_sequences_budgets_,
        kv_cache_manager_->get_swap_block_transfer_infos());
  } else {
    // No pipeline has been selected before the first request arrives.
    CHECK(running_requests_.empty());
    CHECK(running_sequences_.empty());
    batches = RecBatchGroup(static_cast<size_t>(options_.dp_size()),
                            BatchInputType::SEQUENCE);
  }

  // update metrics before returning
  if (std::any_of(batches.begin(), batches.end(), [](const RecBatch& batch) {
        return !batch.empty();
      })) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
    kv_cache_manager_->transfer_blocks(batches);
  } else {
    kv_cache_manager_->transfer_blocks();
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests, prefill_queue_->size());

  GAUGE_ADD(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            kv_cache_manager_->num_blocks_in_prefix_cache().size());
  GAUGE_SET(num_free_blocks, kv_cache_manager_->num_free_blocks().size());
  GAUGE_SET(num_used_blocks, kv_cache_manager_->num_used_blocks().size());

  return batches;
}

ScheduleResult FixedStepsScheduler::schedule_request(
    const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  ScheduleResult result;
  while (true) {
    const SchedulerNotifier::Snapshot snapshot = notifier_->snapshot();
    if (snapshot.stop_requested) {
      return result;
    }

    result.batches = prepare_rec_batch();
    bool all_empty = std::all_of(
        result.batches.begin(),
        result.batches.end(),
        [](const RecBatch& one_batch) { return one_batch.empty(); });
    if (!all_empty) {
      // Move running_requests_ and running_sequences_ into result
      result.requests = std::move(running_requests_);
      result.sequences = std::move(running_sequences_);
      return result;
    }
    const auto now = absl::Now();
    if (now >= deadline) {
      break;
    }
    const absl::Duration remaining = deadline - now;
    // Idle schedulers wait until the deadline or the next notification. Keep
    // short retries while prefetches or local admissions still need progress.
    const bool has_pending_work =
        prefetching_requests_.load(std::memory_order_relaxed) > 0 ||
        !prefill_queue_->empty();
    const absl::Duration wait_duration =
        has_pending_work ? std::min(absl::Milliseconds(1), remaining)
                         : remaining;
    if (notifier_->wait_for(snapshot.generation,
                            absl::ToChronoNanoseconds(wait_duration))) {
      break;
    }
  }
  // return empty result
  return result;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void FixedStepsScheduler::step(const absl::Duration& timeout) {
  if (!options_.enable_schedule_overlap()) {
    // get a new batch of requests
    ScheduleResult result = schedule_request(timeout);
    bool all_empty = std::all_of(
        result.batches.begin(),
        result.batches.end(),
        [](const RecBatch& one_batch) { return one_batch.empty(); });
    if (all_empty) {
      return;
    }

    // Submit task to thread pool for asynchronous execution
    // After engine_->step() completes, process finished/cancelled requests
    auto function = [this,
                     batches = std::move(result.batches),
                     requests = std::move(result.requests),
                     sequences = std::move(result.sequences)]() mutable {
      engine_->step(batches);

      // After step completes, check and process finished/cancelled requests
      std::vector<std::shared_ptr<Request>> finished_requests;
      for (auto& request : requests) {
        if (request) {
          request->update_connection_status();
          if (request->finished() || request->cancelled()) {
            kv_cache_manager_->deallocate(request.get());
            finished_requests.emplace_back(request);
          }
        }
      }

      // Process finished requests
      if (!finished_requests.empty()) {
        output_dispatcher_->process_completed_requests(finished_requests);
      }

      if (options_.rec_worker_max_concurrency() > 1) {
        step_semaphore_.release();
      }
    };

    if (options_.rec_worker_max_concurrency() > 1) {
      step_semaphore_.acquire();
      step_threadpool_->schedule(std::move(function));
    } else {
      function();
    }

    // Return immediately to allow the next step() call to execute in parallel
  } else {
    LOG(ERROR) << "FixedStepsScheduler::step() not supported with "
                  "enable_schedule_overlap";
  }
}

void FixedStepsScheduler::generate() {
  while (num_pending_requests() > 0 || request_queue_.size() > 0 ||
         has_pending_prefetch()) {
    step(absl::Milliseconds(50));
  }
  output_dispatcher_->wait_completion();
}

// Pipeline implementations
std::optional<KVCacheManager::KVCacheReservation>
FixedStepsScheduler::LlmRecSchedulerPipeline::kv_cache_reservation(
    Sequence* sequence) const {
  return FixedStepsScheduler::make_kv_cache_reservation(
      sequence, sequence->stopping_checker()->get_max_generated_tokens());
}

std::optional<KVCacheManager::KVCacheReservation>
FixedStepsScheduler::OneRecXAttentionSchedulerPipeline::kv_cache_reservation(
    Sequence* sequence) const {
  size_t max_generated_tokens =
      ::xllm::RecConfig::get_instance().max_decode_rounds() > 0
          ? static_cast<size_t>(
                ::xllm::RecConfig::get_instance().max_decode_rounds())
          : kRecDecodeSteps;
  if (const auto* stopping_checker = sequence->stopping_checker()) {
    max_generated_tokens = std::max(
        max_generated_tokens, stopping_checker->get_max_generated_tokens());
  }

  return FixedStepsScheduler::make_kv_cache_reservation(sequence,
                                                        max_generated_tokens);
}

std::optional<KVCacheManager::KVCacheReservation>
FixedStepsScheduler::make_kv_cache_reservation(Sequence* sequence,
                                               size_t max_generated_tokens) {
  const size_t num_tokens = sequence->num_tokens();
  if (std::numeric_limits<size_t>::max() - num_tokens < max_generated_tokens) {
    return std::nullopt;
  }
  return KVCacheManager::KVCacheReservation{sequence,
                                            num_tokens + max_generated_tokens};
}

std::unique_ptr<FixedStepsScheduler::SchedulerPipeline>
FixedStepsScheduler::create_scheduler_pipeline(RecType rec_type,
                                               bool is_rec_multi_round) {
  if (is_rec_multi_round) {
    return std::make_unique<RecMultiRoundSchedulerPipeline>();
  }
  if (rec_type == RecType::kOneRec && is_onerec_xattention_mode()) {
    return std::make_unique<OneRecXAttentionSchedulerPipeline>();
  }
  if (rec_type == RecType::kLlmRec) {
    return std::make_unique<LlmRecSchedulerPipeline>();
  }
  return std::make_unique<OneRecSchedulerPipeline>();
}

}  // namespace xllm

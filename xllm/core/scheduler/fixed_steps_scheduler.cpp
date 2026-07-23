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

#include "fixed_steps_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <folly/Unit.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/rec_type.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/rec_model_utils.h"

namespace xllm {

FixedStepsScheduler::FixedStepsScheduler(Engine* engine, const Options& options)
    : ContinuousScheduler(engine, options),
      step_semaphore_(
          static_cast<std::ptrdiff_t>(options.rec_worker_max_concurrency())) {
  step_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/static_cast<size_t>(options.rec_worker_max_concurrency()),
      /*cpu_binding=*/false,
      /*pool_name=*/"FixedStepsScheduler.step");
}

bool FixedStepsScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  if (request_queue_.write(request)) {  //.get()
    // take over the ownership of the request
    // request.release();
    return true;
  }
  // queue is full
  return false;
}

void FixedStepsScheduler::handle_prefill_requests(
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
  // they may contian many sequences, so we should check here.
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
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
      CHECK(num_sequences == 1)
          << "Waiting request should have only one sequence.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
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
        if (!scheduler_pipeline_->allocate_kv_cache(kv_cache_manager_,
                                                    prefill_sequence.get())) {
          can_schedule = false;
          blocks_exhausted = true;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        if (requires_kv_cache) {
          kv_cache_manager_->deallocate(seq);
        }
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
      decode_queue_->empty()) {
    LOG(ERROR)
        << "Request prompt is too long, no enough budget/memory to schedule "
           "a single sequence.";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(prefill_queue_->top());
    prefill_queue_->pop_top();
    // block_manager_->release_blocks_for(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough budget to schedule single sequence."});
  }
}

std::vector<Batch> FixedStepsScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to prefill_queue_
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

  // Drain the request prefetched by the blocking wait in schedule_request().
  if (prefetched_request_) {
    propagate_request(prefetched_request_);
    prefetched_request_.reset();
  }

  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
  while (request_queue_.read(request)) {
    propagate_request(request);
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

  // Lazy initialize pipeline before handle_prefill_requests
  // Because handle_prefill_requests accesses
  // scheduler_pipeline_->requires_kv_cache(), we need to initialize it earlier.
  // Initialize from prefill_queue_ since running_requests_ was just
  // cleared.
  if (!scheduler_pipeline_ && !prefill_queue_->empty()) {
    const std::shared_ptr<Request>& sample_request = prefill_queue_->top();
    auto rec_type = sample_request->state().rec_type;
    bool is_rec_multi_round =
        (rec_type == RecType::kLlmRec) && is_rec_multi_round_mode();
    scheduler_pipeline_ =
        create_scheduler_pipeline(rec_type, is_rec_multi_round);
  }

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;

  handle_prefill_requests(
      remaining_token_budget, remaining_seq_budget, finished_requests);

  // only forward once, no decode requests
  // handle_decode_requests(
  //     remaining_token_budget, remaining_seq_budget, num_preempted_requests);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto* batch_factory = BatchFactory::get_instance(options_.dp_size());

  // Use pipeline to create batches
  std::vector<Batch> batches;
  if (scheduler_pipeline_) {
    batches = scheduler_pipeline_->create_batches(*this, batch_factory);
  } else {
    // Fallback for empty requests
    batches = batch_factory->create_rec_batches(
        running_requests_,
        running_sequences_,
        running_sequences_budgets_,
        kv_cache_manager_->get_swap_block_transfer_infos());
  }

  // update metrics before returning
  if (!batches[0].empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
    kv_cache_manager_->transfer_blocks(batches);
  } else {
    kv_cache_manager_->transfer_blocks();
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            prefill_queue_->size() + decode_queue_->size());

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
    result.batches = prepare_batch();
    bool all_empty =
        std::all_of(result.batches.begin(),
                    result.batches.end(),
                    [](const Batch& one_batch) { return one_batch.empty(); });
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
    // Event-driven wait instead of fixed-interval busy polling: block on the
    // request queue until a new request arrives or the deadline is reached.
    // This wakes up immediately on arrival, avoiding the extra latency and CPU
    // spinning of a fixed sleep under high concurrency. The prefetched request
    // is consumed by the next prepare_batch() call.
    std::shared_ptr<Request> request;
    const auto wait_deadline = std::chrono::steady_clock::now() +
                               absl::ToChronoNanoseconds(deadline - now);
    if (request_queue_.tryReadUntil(wait_deadline, request)) {
      prefetched_request_ = std::move(request);
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
    bool all_empty =
        std::all_of(result.batches.begin(),
                    result.batches.end(),
                    [](const Batch& one_batch) { return one_batch.empty(); });
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
        response_processor_->process_completed_requests(finished_requests);
      }

      if (options_.rec_worker_max_concurrency() > 1) {
        step_semaphore_.release();
      }
    };

    if (options_.rec_worker_max_concurrency() > 1) {
      step_semaphore_.acquire();
      step_threadpool_->schedule(function);
    } else {
      function();
    }

    // Return immediately to allow the next step() call to execute in parallel
  } else {
    LOG(ERROR) << "FixedStepsScheduler::step() not supported with "
                  "enable_schedule_overlap";
  }
}

// Pipeline implementations
std::vector<Batch> FixedStepsScheduler::LlmRecSchedulerPipeline::create_batches(
    FixedStepsScheduler& scheduler,
    BatchFactory* batch_factory) {
  return batch_factory->create_batches(
      scheduler.running_requests_,
      scheduler.running_sequences_,
      scheduler.running_sequences_budgets_,
      scheduler.kv_cache_manager_->get_swap_block_transfer_infos());
}

bool FixedStepsScheduler::LlmRecSchedulerPipeline::allocate_kv_cache(
    KVCacheManager* kv_cache_manager,
    Sequence* sequence) {
  const size_t num_tokens = sequence->num_tokens();
  const size_t max_generated_tokens =
      sequence->stopping_checker()->get_max_generated_tokens();
  // Overflow check to prevent undersized KV cache allocation
  if (std::numeric_limits<size_t>::max() - num_tokens < max_generated_tokens) {
    LOG(ERROR) << "Integer overflow detected in KV cache allocation";
    return false;
  }
  return kv_cache_manager->allocate(sequence,
                                    num_tokens + max_generated_tokens);
}

std::vector<Batch> FixedStepsScheduler::OneRecSchedulerPipeline::create_batches(
    FixedStepsScheduler& scheduler,
    BatchFactory* batch_factory) {
  return batch_factory->create_rec_batches(
      scheduler.running_requests_,
      scheduler.running_sequences_,
      scheduler.running_sequences_budgets_,
      scheduler.kv_cache_manager_->get_swap_block_transfer_infos());
}

std::vector<Batch>
FixedStepsScheduler::OneRecXAttentionSchedulerPipeline::create_batches(
    FixedStepsScheduler& scheduler,
    BatchFactory* batch_factory) {
  return batch_factory->create_rec_batches(
      scheduler.running_requests_,
      scheduler.running_sequences_,
      scheduler.running_sequences_budgets_,
      scheduler.kv_cache_manager_->get_swap_block_transfer_infos());
}

bool FixedStepsScheduler::OneRecXAttentionSchedulerPipeline::allocate_kv_cache(
    KVCacheManager* kv_cache_manager,
    Sequence* sequence) {
  const size_t num_tokens = sequence->num_tokens();
  size_t max_generated_tokens =
      ::xllm::RecConfig::get_instance().max_decode_rounds() > 0
          ? static_cast<size_t>(
                ::xllm::RecConfig::get_instance().max_decode_rounds())
          : kRecDecodeSteps;
  if (const auto* stopping_checker = sequence->stopping_checker()) {
    max_generated_tokens = std::max(
        max_generated_tokens, stopping_checker->get_max_generated_tokens());
  }
  if (std::numeric_limits<size_t>::max() - num_tokens < max_generated_tokens) {
    LOG(ERROR) << "Integer overflow detected in OneRec xattention KV cache "
                  "allocation";
    return false;
  }
  return kv_cache_manager->allocate(sequence,
                                    num_tokens + max_generated_tokens);
}

std::vector<Batch>
FixedStepsScheduler::RecMultiRoundSchedulerPipeline::create_batches(
    FixedStepsScheduler& scheduler,
    BatchFactory* batch_factory) {
  return batch_factory->create_batches(
      scheduler.running_requests_,
      scheduler.running_sequences_,
      scheduler.running_sequences_budgets_,
      scheduler.kv_cache_manager_->get_swap_block_transfer_infos());
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

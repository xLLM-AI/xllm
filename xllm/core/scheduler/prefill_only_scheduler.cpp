/* Copyright 2025-2026 The xLLM Authors.
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

#include "scheduler/prefill_only_scheduler.h"

#include <limits>

#include "common/metrics.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

// Linear-state models (Qwen3.5 GDN) must size the KV allocation by the full
// prompt: a prefix-cache hit clamps the committed KV position down to the
// linear-recoverable length, so a later chunk would advance
// kv_cache_tokens_num() past a capacity grown only for this step's budget
// (tripping batch_input_builder's current_max_tokens_capacity >= seq_len
// invariant). Pure-KV models keep the original per-step sizing, so their
// scheduling / memory behavior is unchanged.
size_t get_prefill_allocate_tokens(Sequence* sequence,
                                   size_t step_tokens,
                                   bool is_linear_state_prefix_cache) {
  if (is_linear_state_prefix_cache) {
    return sequence->num_tokens();
  }
  return sequence->kv_state().kv_cache_tokens_num() + step_tokens;
}

}  // namespace

PrefillOnlyScheduler::PrefillOnlyScheduler(Engine* engine,
                                           const Options& options)
    : ContinuousScheduler(engine, options) {}

PrefillOnlyScheduler::~PrefillOnlyScheduler() {
  // release all requests in the priority queue
  while (!waiting_priority_queue_->empty()) {
    waiting_priority_queue_->pop_top();
  }

  // release all requests in the running priority queue
  while (!running_queue_->empty()) {
    running_queue_->pop_top();
  }
}

void PrefillOnlyScheduler::handle_prefill_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    RequestPriorityQueue* waiting_priority_queue,
    size_t& num_online_prefill_preempt_offline_requests,
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

  while (!waiting_priority_queue->empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 && latency_budget > estimate_latency) {
    if (kv_cache_manager_->kv_cache_utilization() >=
        ::xllm::SchedulerConfig::get_instance()
            .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(waiting_priority_queue->top());
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      waiting_priority_queue->pop_top();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    if (!kv_cache_manager_->update_prefetch_result(
            request, options_.prefetch_timeout())) {
      waiting_priority_queue->pop_top();
      waiting_priority_queue->push(request);
      continue;
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    // TODO: can refactor like handle_decode otherwise request with multiple
    // long sequences may stuck when n>1
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    size_t allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      // Match shared prefix blocks before sizing this step's allocation. The
      // match advances kv_cache_tokens_num() to the cached prefix length, so
      // both the per-step token budget below and get_prefill_allocate_tokens()
      // observe the post-match kv state. Without it, a new sequence whose
      // per-step budget is clamped below its cached prefix would under-request:
      // the prefix match still bumps kv inside allocate(), but the requested
      // token count stays below the matched prefix, so capacity never grows to
      // cover kv_cache_tokens_num() + budget (batch_input_builder invariant).
      if (prefill_sequence->kv_state().num_blocks(BlockType::KV) == 0) {
        kv_cache_manager_->allocate_shared(prefill_sequence.get());
      }

      // FIXME: use actual num_tokens to handle
      // Currently overestimating the number of tokens actually processed when
      // enable prefix cache
      // size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      size_t num_tokens = std::min(prefill_sequence->num_need_compute_tokens(),
                                   remaining_token_budget);
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // preempt offline decode
      const size_t max_handle_num_tokens = get_prefill_allocate_tokens(
          prefill_sequence.get(),
          num_tokens,
          /*is_linear_state_prefix_cache=*/
          has_linear_attention_layers_ && enable_prefix_cache_);
      if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                       max_handle_num_tokens)) {
        can_schedule = false;
        if (options_.enable_online_preempt_offline() && !request->offline() &&
            !running_queue_offline_->empty()) {
          size_t num_request_to_evict = 0;
          // according to the prefill_sequence num tokens to check if can
          // allocate blocks for it through evict

          bool enough_to_evict =
              check_if_enough_to_evict(running_queue_offline_.get(),
                                       prefill_sequence.get(),
                                       max_handle_num_tokens,
                                       num_request_to_evict);
          if (enough_to_evict) {
            for (size_t i = 0; i < num_request_to_evict; ++i) {
              std::shared_ptr<Request> request_to_preempt =
                  running_queue_offline_->back();
              ++num_online_prefill_preempt_offline_requests;
              kv_cache_manager_->deallocate(request_to_preempt.get());
              running_queue_offline_->pop_back();
              // add preemptable request to waiting priority queue
              // TO IMPROVE?: not process this offline request in current batch
              request_to_preempt->set_preempted();
              waiting_priority_queue_offline_->push(request_to_preempt);
            }
            if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                             max_handle_num_tokens)) {
              LOG(ERROR) << "Should be able to allocate after preempting "
                         << num_request_to_evict
                         << " offline requests, but failed.";
              can_schedule = false;
            } else {
              can_schedule = true;
            }
          }
        }
        if (!can_schedule) {
          // release shared prefix blocks
          kv_cache_manager_->deallocate(prefill_sequence.get());
          blocks_exhausted = true;
          break;
        }
      }

      // OPTIMIZE for multi-slo requests
      // for prefill requests, check latency after prefix cache match
      double seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        seq_estimate_latency =
            profile_manager_->predict_step_time(prefill_sequence.get(), false);
        if (estimate_latency + allocated_estimate_latency +
                seq_estimate_latency >
            latency_budget) {
          // release shared prefix blocks
          kv_cache_manager_->deallocate(prefill_sequence.get());
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_->deallocate(seq);
      }
      break;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    estimate_latency += allocated_estimate_latency;
    waiting_priority_queue->pop_top();
    running_requests_.emplace_back(request);
    request->record_num_prefix_cache_tokens();
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget);
  }
  // maybe can pre-compute if prompt beyond length
  if (running_sequences_.empty() && !waiting_priority_queue->empty() &&
      running_queue_->empty()) {
    std::shared_ptr<Request> request(waiting_priority_queue->top());
    waiting_priority_queue->pop_top();
    kv_cache_manager_->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budegt.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

void PrefillOnlyScheduler::handle_last_step_prefill_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& last_step_prefill_requests,
    size_t& num_online_prefill_preempt_offline_requests,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  bool budget_exhausted = false;
  bool blocks_exhausted = false;

  size_t req_idx = 0;
  while (req_idx < last_step_prefill_requests.size() &&
         remaining_seq_budget > 0 && remaining_token_budget > 0 &&
         latency_budget > estimate_latency) {
    if (kv_cache_manager_->kv_cache_utilization() >=
        ::xllm::SchedulerConfig::get_instance()
            .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(last_step_prefill_requests[req_idx++]);
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1 || num_sequences == request->best_of())
          << "Waiting request should have either 1 or best_of("
          << request->best_of() << ") sequences, got " << num_sequences;
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    // TODO: can refactor like handle_decode otherwise request with multiple
    // long sequences may stuck when n>1
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    size_t allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      // FIXME: use actual num_tokens to handle
      // Currently overestimating the number of tokens actually processed when
      // enable prefix cache
      // size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      size_t num_tokens = std::min(prefill_sequence->num_need_compute_tokens(),
                                   remaining_token_budget);
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // preempt offline decode
      const size_t max_handle_num_tokens = get_prefill_allocate_tokens(
          prefill_sequence.get(),
          num_tokens,
          /*is_linear_state_prefix_cache=*/
          has_linear_attention_layers_ && enable_prefix_cache_);
      if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                       max_handle_num_tokens)) {
        can_schedule = false;
        if (options_.enable_online_preempt_offline() && !request->offline() &&
            !running_queue_offline_->empty()) {
          size_t num_request_to_evict = 0;
          // according to the prefill_sequence num tokens to check if can
          // allocate blocks for it through evict

          bool enough_to_evict =
              check_if_enough_to_evict(running_queue_offline_.get(),
                                       prefill_sequence.get(),
                                       max_handle_num_tokens,
                                       num_request_to_evict);
          if (enough_to_evict) {
            for (size_t i = 0; i < num_request_to_evict; ++i) {
              std::shared_ptr<Request> request_to_preempt =
                  running_queue_offline_->back();
              ++num_online_prefill_preempt_offline_requests;
              kv_cache_manager_->deallocate(request_to_preempt.get());
              running_queue_offline_->pop_back();
              // add preemptable request to waiting priority queue
              // TO IMPROVE?: not process this offline request in current batch
              request_to_preempt->set_preempted();
              waiting_priority_queue_offline_->push(request_to_preempt);
            }
            if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                             max_handle_num_tokens)) {
              LOG(ERROR) << "Should be able to allocate after preempting "
                         << num_request_to_evict
                         << " offline requests, but failed.";
              can_schedule = false;
            } else {
              can_schedule = true;
            }
          }
        }
        if (!can_schedule) {
          // release shared prefix blocks
          kv_cache_manager_->deallocate(prefill_sequence.get());
          blocks_exhausted = true;
          break;
        }
      }

      // OPTIMIZE for multi-slo requests
      // for prefill requests, check latency after prefix cache match
      double seq_estimate_latency = 0;
      if (options_.enable_latency_aware_schedule()) {
        seq_estimate_latency =
            profile_manager_->predict_step_time(prefill_sequence.get(), false);
        if (estimate_latency + allocated_estimate_latency +
                seq_estimate_latency >
            latency_budget) {
          // release shared prefix blocks
          kv_cache_manager_->deallocate(prefill_sequence.get());
          can_schedule = false;
          budget_exhausted = true;
          break;
        }
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
      allocated_estimate_latency += seq_estimate_latency;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_->deallocate(seq);
      }
      break;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    estimate_latency += allocated_estimate_latency;
    // waiting_priority_queue.pop();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
    cache_in_batch_prefix(prefill_sequences, prefill_sequences_budget);
  }
  // maybe can pre-compute if prompt beyond length
  if (running_sequences_.empty() && !last_step_prefill_requests.empty() &&
      running_queue_->empty()) {
    std::shared_ptr<Request> request(last_step_prefill_requests.front());
    last_step_prefill_requests.erase(last_step_prefill_requests.begin());
    kv_cache_manager_->deallocate(request.get());
    if (blocks_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                    "a single sequence.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough memory to schedule single sequence"});
    } else if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, no enough budget to schedule "
                    "a single sequence. Please set a larger budegt.";
      // no enough memory to schedule single sequence, just finish the request
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough budget to schedule single sequence."});
    } else {
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

std::vector<Batch> PrefillOnlyScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  // Include those requests that are preempted by others.
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
  while (request_queue_.read(request)) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      if (request->offline()) {
        waiting_priority_queue_offline_->push(request);
      } else {
        waiting_priority_queue_->push(request);
      }
    } else {
      // request from prefill instance in disagge pd mode.
      running_requests_.emplace_back(request);
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
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  std::vector<std::shared_ptr<Request>> last_step_prefill_requests;

  if (options_.priority_strategy() == "fcfs") {
    if (last_step_prefill_) {
      // insert all requests to the back of running_queue_
      // 1. last step is prefill step:
      // new prefill has high priority, but these requests has lower priority
      // then existed requests in running_queue_ in decoding stage.
      // so we need to push them to the back of running_queue_.
      for (auto it = running_requests_.cbegin(); it != running_requests_.cend();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->is_chunked_prefill_stage()) {
          last_step_prefill_requests.emplace_back(*it);
        } else {
          if ((*it)->offline()) {
            running_queue_offline_->push(*it, last_step_prefill_);
          } else {
            running_queue_->push(*it, last_step_prefill_);
          }
        }
      }
    } else {
      // insert all requests to the front of running_queue_
      // 2. last step is decode step:
      // We need to traverse running_requests_ array in reverse order.
      // Because there may be some unexecuted requests with
      // lower priorities remaining in the running_queue_.
      // For the requests in running_requests_,
      // their priorities are all higher than those of the
      // remaining requests. Therefore, the `push_front`
      // method needs to be used.
      //
      for (auto it = running_requests_.crbegin();
           it != running_requests_.crend();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->offline()) {
          running_queue_offline_->push(*it, last_step_prefill_);
        } else {
          running_queue_->push(*it, last_step_prefill_);
        }
      }
    }
  } else {
    // directly push running requests to the priority queue
    for (auto it = running_requests_.cbegin(); it != running_requests_.cend();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      if ((*it)->is_chunked_prefill_stage()) {
        last_step_prefill_requests.emplace_back(*it);
      } else {
        if ((*it)->offline()) {
          running_queue_offline_->push(*it);
        } else {
          running_queue_->push(*it);
        }
      }
    }
  }

  // clear previous batch
  last_step_prefill_ = false;
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // maintain estimate_latency for current batch for support requests with
  // different ttft. TO IMPROVE: use min remaining time (i.e. slo -
  // elapsed_time) of the reuquest in current decode queue to replace current
  // latency_budget.
  double latency_budget = options_.max_global_ttft_ms();
  double estimate_latency = 0;
  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;
  size_t num_offline_decode_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_online_requests = 0;
  size_t num_online_prefill_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_offline_requests = 0;

  // 1. handle last step prefill requests
  // try to finish prefill requests as soon as fast as possible
  if (!last_step_prefill_requests.empty()) {
    handle_last_step_prefill_requests(
        latency_budget,
        estimate_latency,
        remaining_token_budget,
        remaining_seq_budget,
        last_step_prefill_requests,
        num_online_prefill_preempt_offline_requests,
        finished_requests);
  }
  // 2. handle prefill requests
  // try to schedule prefill request if have remaining budget
  // TO IMPROVE?: handle online decode request before prefill offline request
  handle_prefill_requests(latency_budget,
                          estimate_latency,
                          remaining_token_budget,
                          remaining_seq_budget,
                          waiting_priority_queue_.get(),
                          num_online_prefill_preempt_offline_requests,
                          finished_requests);
  handle_prefill_requests(latency_budget,
                          estimate_latency,
                          remaining_token_budget,
                          remaining_seq_budget,
                          waiting_priority_queue_offline_.get(),
                          num_online_prefill_preempt_offline_requests,
                          finished_requests);

  // 3. handle decode requests
  // no prefill request, schedule the decode requests in the running priority
  // queue
  if (running_sequences_.empty()) {
    latency_budget = options_.max_global_tpot_ms();
    handle_decode_requests(latency_budget,
                           estimate_latency,
                           remaining_token_budget,
                           remaining_seq_budget,
                           num_offline_decode_preempt_offline_requests,
                           num_online_decode_preempt_online_requests,
                           num_online_decode_preempt_offline_requests,
                           running_queue_.get());
    handle_decode_requests(latency_budget,
                           estimate_latency,
                           remaining_token_budget,
                           remaining_seq_budget,
                           num_offline_decode_preempt_offline_requests,
                           num_online_decode_preempt_online_requests,
                           num_online_decode_preempt_offline_requests,
                           running_queue_offline_.get());
  }

  num_preempted_requests = num_offline_decode_preempt_offline_requests +
                           num_online_decode_preempt_online_requests +
                           num_online_decode_preempt_offline_requests +
                           num_online_prefill_preempt_offline_requests;
  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(running_requests_,
                           running_sequences_,
                           running_sequences_budgets_,
                           kv_cache_manager_->get_swap_block_transfer_infos());

  bool is_batches_empty =
      (std::all_of(batches.begin(), batches.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      }));
  if (!is_batches_empty) {
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
            waiting_priority_queue_->size() + running_queue_->size());

  GAUGE_ADD(num_preempted_requests, num_preempted_requests);
  GAUGE_ADD(num_offline_decode_preempt_offline_requests,
            num_offline_decode_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_online_requests,
            num_online_decode_preempt_online_requests);
  GAUGE_ADD(num_online_prefill_preempt_offline_requests,
            num_online_prefill_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_offline_requests,
            num_online_decode_preempt_offline_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(kv_cache_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(kv_cache_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(kv_cache_manager_->num_used_blocks()));
  return batches;
}

}  // namespace xllm

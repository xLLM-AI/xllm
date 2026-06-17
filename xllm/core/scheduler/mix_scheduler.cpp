/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "scheduler/mix_scheduler.h"

#include <algorithm>
#include <limits>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

MixScheduler::MixScheduler(Engine* engine, const Options& options)
    : ChunkedPrefillScheduler(engine, options) {}

MixScheduler::~MixScheduler() {
  // release all requests in the running priority queue
  while (!running_queue_.empty()) {
    running_queue_.pop_front();
  }
}

//----------------------------------

void MixScheduler::get_latency_budget_and_request_order(
    std::list<std::shared_ptr<Request>>& running_queue,
    double& latency_budget) {
  // update request metrcis
  for (auto& request : running_queue) {
    auto& sequence = request->sequences()[0];
    sequence->set_estimated_latency(
        profile_manager_->predict_step_time(sequence.get(), false));
    request->set_elapsed_time_ms();
    request->set_deadline_ms();
    request->set_starved(false);
  }

  auto constant_overhead = profile_manager_->get_constant_overhead();
  double total_exec_time = 0.0;

  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();
  for (auto it = running_queue.cbegin(); it != running_queue.cend(); it++) {
    const auto request = *it;
    const auto& sequence = request->sequences()[0];
    auto remaining_time = request->get_remaining_time();
    total_exec_time += sequence->estimated_latency();
    if (request->tpot_slo_ms() < min_tpot) {
      min_tpot = static_cast<int32_t>(request->tpot_slo_ms());
    }

    if (remaining_time < sequence->estimated_latency() + constant_overhead) {
      // Currently, overdue and urgent requests are being handled together.
      continue;
    }
    if (remaining_time < min_remaining_time) {
      min_remaining_time = static_cast<int32_t>(remaining_time);
    }
  }
  // determine latency budget
  // int32_t threshold = static_cast<int32_t>(4 * constant_overhead);
  int32_t latency_budget_threshold = static_cast<int32_t>(0.65 * min_tpot);
  latency_budget = std::max(min_remaining_time, latency_budget_threshold);

  const double lambda =
      ::xllm::SchedulerConfig::get_instance().aggressive_coeff();
  double load_judge_func =
      total_exec_time * latency_budget / (latency_budget - constant_overhead);
  for (auto& request : running_queue) {  // determine urgency
    auto& sequence = request->sequences()[0];

    if (::xllm::SchedulerConfig::get_instance().enable_starve_prevent()) {
      // avoid overly starvation, threshold can be tuned
      const int32_t starve_unit_time = sequence->is_prefill_stage()
                                           ? -request->ttft_slo_ms()
                                           : -request->tpot_slo_ms();
      const int32_t starve_time_threshold = static_cast<int32_t>(
          ::xllm::SchedulerConfig::get_instance().starve_threshold() *
          starve_unit_time);
      if (request->get_remaining_time() < starve_time_threshold) {
        request->set_starved(true);
      }
    }

    if (request->get_remaining_time() < lambda * load_judge_func) {
      request->set_urgency(Urgency::URGENT);
    } else {
      request->set_urgency(Urgency::NORMAL);
    }
  }

  // sort, should be urgency_density now
  auto priority_strategy = options_.priority_strategy();
  // not specifically configure priority strategy, use optimized priority
  // strategy for mix_scheduler
  if (priority_strategy == "fcfs") {
    priority_strategy = "urgency_density";
  }
  running_queue.sort(create_comparator(priority_strategy, true));
}

size_t MixScheduler::get_needed_copy_block_num(
    std::vector<std::shared_ptr<Request>>& req_vec,
    std::vector<size_t>& req_copy_block_num_vec,
    double max_h2d_transfer_time,
    double min_total_exec_time,
    size_t max_h2d_block_num) {
  auto block_size = kv_cache_manager_->block_size();
  size_t total_needed_copy_blocks = max_h2d_block_num;
  double total_exec_time = min_total_exec_time;
  double h2d_transfer_time = max_h2d_transfer_time;
  CHECK_GT(h2d_transfer_time, total_exec_time);
  size_t index = req_vec.size() - 1;
  for (auto it = req_vec.rbegin(); it != req_vec.rend(); it++, index--) {
    auto request = *it;
    auto& sequence = request->sequences()[0];
    total_needed_copy_blocks -= req_copy_block_num_vec[index];
    total_exec_time -=
        profile_manager_->predict_step_time(sequence.get(), false);
    double cur_seq_max_exec_time = profile_manager_->predict_step_time(
        sequence->num_tokens(),
        sequence->kv_state().kv_cache_tokens_num(),
        false);
    total_exec_time += cur_seq_max_exec_time;
    h2d_transfer_time -= profile_manager_->predict_copy_blocks_time(
        req_copy_block_num_vec[index], false);

    if (h2d_transfer_time < total_exec_time) {
      // binary search find the split point [left,right)
      double base_total_exec_time = total_exec_time - cur_seq_max_exec_time;
      size_t left = 0;
      size_t right = req_copy_block_num_vec[index] + 1;
      double min_latency = std::numeric_limits<double>::max();
      while (left < right) {
        size_t mid = left + (right - left) / 2;
        double cur_seq_h2d_time =
            profile_manager_->predict_copy_blocks_time(mid, false);
        size_t kv_cache_tokens_num =
            mid == 0
                ? sequence->kv_state().kv_cache_tokens_num()
                : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                   mid) *
                      block_size;
        double cur_seq_exec_time = profile_manager_->predict_step_time(
            sequence->num_tokens(), kv_cache_tokens_num, false);
        if (h2d_transfer_time + cur_seq_h2d_time <
            base_total_exec_time + cur_seq_exec_time) {
          left = mid + 1;
          min_latency = std::max(h2d_transfer_time + cur_seq_h2d_time,
                                 base_total_exec_time + cur_seq_exec_time);
        } else {
          right = mid;
        }
      }
      // check if `left` is the best split point
      size_t needed_copy_blocks = left - 1;
      if (left <= req_copy_block_num_vec[index]) {
        double cur_seq_h2d_time =
            profile_manager_->predict_copy_blocks_time(left, false);
        size_t kv_cache_tokens_num =
            left == 0
                ? sequence->kv_state().kv_cache_tokens_num()
                : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                   left) *
                      block_size;
        double cur_seq_exec_time = profile_manager_->predict_step_time(
            sequence->num_tokens(), kv_cache_tokens_num, false);
        double candidate_latency =
            std::max(h2d_transfer_time + cur_seq_h2d_time,
                     base_total_exec_time + cur_seq_exec_time);
        if (min_latency > candidate_latency) {
          needed_copy_blocks = left;
        }
      }

      total_needed_copy_blocks += needed_copy_blocks;
      break;
    }
  }
  return total_needed_copy_blocks;
}

size_t MixScheduler::get_max_copy_block_num(
    std::list<std::shared_ptr<Request>>& running_queue,
    double& latency_budget) {
  double min_total_exec_time = profile_manager_->get_constant_overhead();
  size_t max_h2d_block_num = 0;
  auto block_size = kv_cache_manager_->block_size();
  std::vector<size_t> req_copy_block_num_vec;
  std::vector<std::shared_ptr<Request>> req_vec;
  for (auto& request : running_queue) {
    auto& sequence = request->sequences()[0];
    // use kv cache tokens in both device and host to estimate total batch
    // latency to ensure copy overhead can be hidden
    min_total_exec_time +=
        profile_manager_->predict_step_time(sequence.get(), false);

    size_t host_blocks_num =
        sequence->host_kv_state().kv_cache_tokens_num() / block_size;
    size_t device_blocks_num =
        sequence->kv_state().kv_cache_tokens_num() / block_size;
    size_t cur_step_copy_blocks = host_blocks_num > device_blocks_num
                                      ? host_blocks_num - device_blocks_num
                                      : 0;
    max_h2d_block_num += cur_step_copy_blocks;
    if (cur_step_copy_blocks > 0) {
      req_copy_block_num_vec.push_back(cur_step_copy_blocks);
      req_vec.push_back(request);
    }
  }
  size_t max_copy_block_num = std::numeric_limits<int32_t>::max();
  if (min_total_exec_time >= latency_budget) {
    // case 1: use latency budget to determine copy total blocks num
    max_copy_block_num =
        profile_manager_->get_max_copy_block_num(latency_budget);
  } else {
    double max_h2d_transfer_time =
        profile_manager_->predict_copy_blocks_time(max_h2d_block_num);
    if (max_h2d_transfer_time > min_total_exec_time) {
      // case2: compute to determine need copy total blocks num
      max_copy_block_num = get_needed_copy_block_num(req_vec,
                                                     req_copy_block_num_vec,
                                                     max_h2d_transfer_time,
                                                     min_total_exec_time,
                                                     max_h2d_block_num);
    } else {
      // case 3: copy all blocks
    }
  }
  return max_copy_block_num;
}

void MixScheduler::handle_running_queue_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_preempted_requests,
    std::vector<Sequence*>& prefill_stage_sequences,
    std::list<std::shared_ptr<Request>>& running_queue,
    bool& budget_exhausted,
    bool& blocks_exhausted) {
  if (running_queue.empty()) {
    return;
  }

  get_latency_budget_and_request_order(running_queue, latency_budget);

  size_t remaining_copy_blocks_budget =
      (options_.enable_latency_aware_schedule() &&
       ::xllm::KVCacheStoreConfig::get_instance()
           .enable_control_h2d_block_num())
          ? get_max_copy_block_num(running_queue, latency_budget)
          : std::numeric_limits<int32_t>::max();

  std::vector<std::shared_ptr<Request>> preempted_request_vec;
  bool is_preempt_iterator_valid = true;
  auto preempt_iterator = std::prev(running_queue.end());
  while (!running_queue.empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(running_queue.front());
    if (preempt_iterator == running_queue.begin()) {
      is_preempt_iterator_valid = false;
    }

    const size_t num_sequences = request->sequences().size();
    CHECK(num_sequences == 1) << "currently only support one sequence.";

    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    auto constant_overhead = profile_manager_->get_constant_overhead();

    budget_exhausted = false;
    blocks_exhausted = false;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    size_t allocated_copy_blocks = 0;
    for (auto& sequence : request->sequences()) {
      // skip finished sequence.
      if (sequence->finished()) {
        continue;
      }

      // support kv cache swapping between host and device and try to overlap
      // the computation and copy overhead.
      auto block_size = kv_cache_manager_->block_size();
      size_t host_blocks_num =
          sequence->host_kv_state().kv_cache_tokens_num() / block_size;
      size_t device_blocks_num =
          sequence->kv_state().kv_cache_tokens_num() / block_size;
      size_t cur_step_copy_blocks = host_blocks_num > device_blocks_num
                                        ? host_blocks_num - device_blocks_num
                                        : 0;
      cur_step_copy_blocks =
          std::min(cur_step_copy_blocks,
                   remaining_copy_blocks_budget - allocated_copy_blocks);
      size_t kv_cache_tokens_num =
          cur_step_copy_blocks == 0
              ? sequence->kv_state().kv_cache_tokens_num()
              : (sequence->kv_state().kv_cache_tokens_num() / block_size +
                 cur_step_copy_blocks) *
                    block_size;

      // for ablation
      // kv_cache_tokens_num = sequence->kv_cache_tokens_num();

      size_t num_tokens = sequence->num_tokens();
      size_t assume_max_tokens = remaining_token_budget - allocated_tokens;

      // use either latency_budget or token_budget
      // use latency_budget
      if (options_.enable_latency_aware_schedule()) {
        // FIXME LATER?: is_prefill_stage currently uses device kv cache tokens
        if (sequence->is_prefill_stage()) {
          assume_max_tokens = get_max_chunk(
              sequence.get(),
              num_tokens,
              kv_cache_tokens_num,
              static_cast<int32_t>(latency_budget - estimate_latency));
          if (assume_max_tokens == kv_cache_tokens_num) {
            budget_exhausted = true;
            break;
          }
          if (assume_max_tokens != num_tokens &&
              (assume_max_tokens - kv_cache_tokens_num) <= 50) {
            // Preventing the creation of too many small chunks (this may not be
            // necessary).
            budget_exhausted = true;
            break;
          }
          allocated_estimate_latency = profile_manager_->predict_step_time(
              assume_max_tokens, kv_cache_tokens_num, false);
          assume_max_tokens -= kv_cache_tokens_num;
        } else {
          assume_max_tokens = 1;
          allocated_estimate_latency = profile_manager_->predict_step_time(
              num_tokens, kv_cache_tokens_num, false);
          if (estimate_latency + allocated_estimate_latency > latency_budget) {
            // slack is too small, even decode is unable to include
            budget_exhausted = true;
            break;
          }
        }
      }
      // use token budget
      else {
        size_t num_tokens_to_handle =
            sequence->is_prefill_stage()
                ? std::min(assume_max_tokens, num_tokens - kv_cache_tokens_num)
                : 1 + min_speculative_tokens_required_;
        if (allocated_seqs + 1 > remaining_seq_budget ||
            allocated_tokens + num_tokens_to_handle > remaining_token_budget) {
          budget_exhausted = true;
          break;
        }
      }

      size_t current_step_handle_tokens = 0;
      // no budget left
      if (!allocate_blocks_for(sequence.get(),
                               assume_max_tokens,
                               kv_cache_tokens_num,
                               cur_step_copy_blocks,
                               &current_step_handle_tokens)) {
        blocks_exhausted = true;
        break;
      }

      // update the allocated tokens for the sequence
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      allocated_copy_blocks += cur_step_copy_blocks;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }
    if (!blocks_exhausted && !budget_exhausted) {
      // remove the request from the priority queue
      running_queue.pop_front();
      // add the request to the batch
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      remaining_copy_blocks_budget -= allocated_copy_blocks;
      estimate_latency += allocated_estimate_latency;
      continue;
    }

    if (budget_exhausted) {
      if (candidate_sequences.empty() && running_sequences_.empty()) {
        LOG(ERROR) << "Request prompt = "
                   << request->sequences()[0]->num_tokens()
                   << " is too long, please set a larger "
                      "max_tokens value via --max_tokens_per_batch.";
        running_queue.pop_front();
        kv_cache_manager_->deallocate(request.get());
        response_processor_->process_failed_request(
            request,
            {StatusCode::RESOURCE_EXHAUSTED,
             "No enough resource to schedule a single sequence"});
      }
      break;
    }

    // memory exhausted, preempt lowest priority request and retry.
    // preemptable_requests_ only contain decoding requests.
    // Maybe improve: for prefill stage sequence, we know how many blocks it
    // needs
    bool find_preempt = false;
    while (is_preempt_iterator_valid &&
           preempt_iterator != running_queue.begin()) {
      std::shared_ptr<Request> request_to_preempt = *preempt_iterator;
      if (request_to_preempt.get() != request.get()) {
        if (request_to_preempt->sequences()[0]
                ->kv_state()
                .kv_cache_tokens_num() != 0) {
          // If KV cache is available, preempt.
          ++num_preempted_requests;
          // TO IMPROVE: kv cache offload to cpu
          kv_cache_manager_->deallocate(request_to_preempt.get());
          auto prev_preempt_iterator = preempt_iterator;
          preempt_iterator--;
          running_queue.erase(prev_preempt_iterator);
          // add preemptable request to waiting priority queue
          request_to_preempt->set_preempted();
          preempted_request_vec.push_back(request_to_preempt);
          find_preempt = true;
          break;
        } else {
          // indicates it is prefill, skip
          preempt_iterator--;
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }
    }
    if (find_preempt) {
      continue;
    }
    // no enough memeory to preempt
    if (candidate_sequences.empty() && running_sequences_.empty()) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
      running_queue.pop_front();
      kv_cache_manager_->deallocate(request.get());
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough resource to schedule a single sequence"});
    }
    blocks_exhausted = true;
    break;
  }

  // push back to running queue
  while (!preempted_request_vec.empty()) {
    std::shared_ptr<Request> request(preempted_request_vec.back());
    running_queue.push_back(request);
    preempted_request_vec.pop_back();
  }
}

std::vector<Batch> MixScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting_priority_queue_
  while (request_queue_.read(request)) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      // both prefill and decode in one queue
      running_queue_.push_back(request);
    } else {
      // request from prefill instance in disagge pd mode.
      // NOTE: running_requests_ keep a batch of requests in running state,
      //   sorted by priority from high to low.
      running_requests_.emplace_back(request);
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
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

  // insert running requests back to the priority queue, iterating from the
  // lowest priority to the highest
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    // finished request is set to nullptr
    if (*it == nullptr) {
      continue;
    }
    handle_running_requests(*it);
    // unified multi priority strategy
    running_queue_.push_back(*it);
  }

  // allocate prefix cache ahead
  for (auto& request : running_queue_) {
    auto& sequence = request->sequences()[0];
    // IMPROVEME LATER: only need to get matched token num in prefix cache,
    // better not allocate shared block before the request is scheduled.
    allocate_shared_blocks_for(sequence.get());
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  double latency_budget = options_.max_global_tpot_ms();

  // constant overhead ahead
  double estimate_latency = profile_manager_->get_constant_overhead();
  // Max tokens be handled in once chunked prefill schedule.
  size_t remaining_token_budget = options_.enable_profile_token_budget()
                                      ? profile_manager_->get_token_budget()
                                      : options_.max_tokens_per_batch();
  size_t remaining_seq_budget = options_.max_seqs_per_batch();
  size_t num_preempted_requests = 0;
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  // keep the requests in prefill stage
  std::vector<Sequence*> prefill_stage_sequences;

  handle_running_queue_requests(latency_budget,
                                estimate_latency,
                                remaining_token_budget,
                                remaining_seq_budget,
                                num_preempted_requests,
                                prefill_stage_sequences,
                                running_queue_,
                                budget_exhausted,
                                blocks_exhausted);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(running_requests_,
                                      running_sequences_,
                                      running_sequences_budgets_);

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
  GAUGE_SET(num_preempted_requests, num_preempted_requests);
  if (num_preempted_requests > 0) {
    LOG(INFO) << "Number of preempted requests in this round: "
              << num_preempted_requests;
  }

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(kv_cache_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(kv_cache_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(kv_cache_manager_->num_used_blocks()));

  return batches;
}

int32_t MixScheduler::get_max_chunk(Sequence* sequence,
                                    size_t num_tokens,
                                    size_t kv_cache_tokens_num,
                                    int32_t latency_budget,
                                    bool use_quadratic_formula) {
  if (num_tokens <= kv_cache_tokens_num) {
    return kv_cache_tokens_num;
  }
  if (profile_manager_->predict_step_time(
          num_tokens, kv_cache_tokens_num, false) <= latency_budget) {
    return num_tokens;
  }
  if (latency_budget <= 0) {
    return kv_cache_tokens_num;
  }
  if (use_quadratic_formula) {
    // use quadratic formula to get root
    return profile_manager_->get_quadratic_root(sequence, latency_budget);
  } else {
    // use binary search

    int32_t left = kv_cache_tokens_num + 1;
    int32_t right = num_tokens + 1;
    // [left, right)
    while (left < right) {
      int32_t mid = left + (right - left) / 2;
      auto predict_time =
          profile_manager_->predict_step_time(mid, kv_cache_tokens_num, false);
      if (predict_time <= latency_budget) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left - 1;
  }
}

bool MixScheduler::if_queue_not_empty() { return !running_queue_.empty(); }

bool MixScheduler::allocate_blocks_for(Sequence* sequence,
                                       size_t token_budget,
                                       size_t kv_cache_tokens_num,
                                       size_t needed_copy_blocks_num,
                                       size_t* current_step_handle_tokens) {
  // token budget should be large enough for one speculative decoding step
  CHECK_GT(token_budget, min_speculative_tokens_required_);

  // already allocate before handle_runing_queue
  // allocate_shared_blocks_for(sequence);

  // number of tokens in the kv cache, which are already processed
  // const size_t kv_cache_tokens_num = sequence->kv_cache_tokens_num();

  // the total number tokens for the sequence can be handled till now.
  // there may some tokens can not be handled once when enable chunked prefill.
  size_t max_handle_num_tokens =
      std::min(kv_cache_tokens_num + token_budget, sequence->num_tokens());

  // speculative decoding specific logic,
  // prefill stage don't need speculative decoding.
  //
  // if in decoding stage
  if (options_.num_speculative_tokens() > 0 &&
      !sequence->is_chunked_prefill_stage() && kv_cache_tokens_num > 0) {
    max_handle_num_tokens += min_speculative_tokens_required_;
  }

  // make sure the sequence proceeds forward
  CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *current_step_handle_tokens = max_handle_num_tokens - kv_cache_tokens_num;
  // allocate blocks for the sequence
  if (::xllm::KVCacheStoreConfig::get_instance().host_blocks_factor() > 1.0) {
    return kv_cache_manager_->allocate(
        sequence, max_handle_num_tokens, needed_copy_blocks_num);
  } else {
    return kv_cache_manager_->allocate(sequence, max_handle_num_tokens);
  }
}

}  // namespace xllm

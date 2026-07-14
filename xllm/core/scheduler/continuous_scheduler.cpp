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

#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/platform.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "scheduler/request_priority_queue.h"
#include "util/utils.h"

namespace xllm {
namespace {

size_t estimate_decode_extra_blocks(Sequence* sequence,
                                    size_t updated_num_tokens,
                                    size_t block_size) {
  const size_t num_blocks = sequence->kv_state().num_blocks(BlockType::KV);
  const size_t num_blocks_needed =
      (updated_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > num_blocks) {
    return num_blocks_needed - num_blocks;
  }

  // Beam swap may still require one extra block when reusing source blocks.
  if (sequence->check_beam_search() &&
      !sequence->kv_state().src_blocks().empty() &&
      sequence->kv_state().need_swap()) {
    return 1;
  }
  return 0;
}

size_t get_sequence_free_blocks_for_rank(KVCacheManager* kv_cache_manager,
                                         int32_t dp_rank) {
  const auto free_blocks = kv_cache_manager->num_free_blocks();
  if (free_blocks.empty()) {
    return 0;
  }
  if (dp_rank >= 0 && static_cast<size_t>(dp_rank) < free_blocks.size()) {
    return free_blocks[dp_rank];
  }
  return util::max(free_blocks);
}

}  // namespace

namespace {

inline size_t maybe_align_cp_prefill_tokens(const Sequence* sequence,
                                            size_t num_tokens,
                                            int32_t cp_size) {
  if (sequence == nullptr || cp_size <= 1 || num_tokens == 0) {
    return num_tokens;
  }
  if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
    return num_tokens;
  }
  if (!sequence->is_prefill_stage()) {
    return num_tokens;
  }
  const size_t alignment = static_cast<size_t>(cp_size) * 2;
  return xllm::util::align_up(num_tokens, alignment);
}

}  // namespace

void CancelRequestQueue::submit(std::shared_ptr<Request> request) {
  std::lock_guard<std::mutex> lock(mutex_);
  requests_.emplace_back(std::move(request));
}

std::vector<std::shared_ptr<Request>> CancelRequestQueue::take_all() {
  std::vector<std::shared_ptr<Request>> requests;
  std::lock_guard<std::mutex> lock(mutex_);
  requests.swap(requests_);
  return requests;
}

ContinuousScheduler::ContinuousScheduler(Engine* engine, const Options& options)
    : options_(options),
      engine_(engine),
      request_queue_(::xllm::RecConfig::get_instance().request_queue_size()) {
  CHECK(engine_ != nullptr);

  kv_cache_manager_ = engine_->block_manager_pool();
  CHECK(kv_cache_manager_ != nullptr);

  enable_prefix_cache_ =
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
  enable_in_batch_prefix_cache_ =
      ::xllm::KVCacheConfig::get_instance().enable_in_batch_prefix_cache();

  last_batch_.resize(options_.dp_size());

  ProfileManager::Options profile_manager_options;
  profile_manager_options.dp_size(options.dp_size())
      .enable_schedule_overlap(options.enable_schedule_overlap())
      .enable_profile_step_time(options.enable_profile_step_time())
      .profile_max_prompt_length(options.profile_max_prompt_length())
      .enable_profile_kv_blocks(options.enable_profile_kv_blocks())
      .max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_global_tpot_ms(options.max_global_tpot_ms())
      .max_global_ttft_ms(options.max_global_ttft_ms())
      .instance_role(options.instance_role().value_or(InstanceRole::DEFAULT))
      .enable_profile_token_budget(options.enable_profile_token_budget());
  profile_manager_ =
      std::make_unique<ProfileManager>(engine, profile_manager_options);

  cancel_request_queue_ = std::make_shared<CancelRequestQueue>();
  response_processor_ = std::make_unique<AsyncResponseProcessor>(
      engine_->tokenizer(),
      options_.instance_role(),
      options_.enable_service_routing(),
      options_.disable_log_stats(),
      [cancel_request_queue =
           cancel_request_queue_](std::shared_ptr<Request> request) {
        cancel_request_queue->submit(std::move(request));
      });
  create_waiting_queue(options);
  create_running_queue(options);
  if (options_.enable_service_routing()) {
    // connect to master service
    xservice_client_ = XServiceClient::get_instance();
    if (!xservice_client_->initialize_done()) {
      LOG(FATAL) << "XServiceClient not init.";
      return;
    }
    xservice_client_->set_scheduler(this);
    if (::xllm::KVCacheConfig::get_instance().enable_xtensor() &&
        !options_.enable_disagg_pd()) {
      xservice_client_->set_engine(engine_);
      engine_->get_cache_info(instance_info_.cluster_ids,
                              instance_info_.addrs,
                              instance_info_.ports);
    }
  }

  instance_info_.name = options_.instance_name().value_or("");
  instance_info_.type = options_.instance_role().value().to_string();
  instance_info_.dp_size = options.dp_size();
  instance_info_.kv_split_size =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();

  if (options_.enable_schedule_overlap()) {
    min_speculative_tokens_required_ = options_.num_speculative_tokens() * 2;
  } else {
    min_speculative_tokens_required_ = options_.num_speculative_tokens();
  }
}

ContinuousScheduler::~ContinuousScheduler() { running_requests_.clear(); }

bool ContinuousScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  kv_cache_manager_->prefetch_from_storage(request);

  if (request_queue_.write(request)) {
    return true;
  }

  return false;
}

void ContinuousScheduler::get_latency_budget_and_request_order(
    RequestPriorityQueue* request_priority_queue,
    double& latency_budget,
    bool for_prefill) {
  if (request_priority_queue == nullptr || request_priority_queue->empty()) {
    return;
  }
  CHECK(profile_manager_ != nullptr);

  const double constant_overhead = profile_manager_->get_constant_overhead();
  double total_exec_time = 0.0;
  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();

  for (auto it = request_priority_queue->begin();
       it != request_priority_queue->end();
       ++it) {
    auto request = *it;
    auto& sequence = request->sequences()[0];
    sequence->set_estimated_latency(
        profile_manager_->predict_step_time(sequence.get(), false));
    request->set_elapsed_time_ms();
    request->set_deadline_ms();
    request->set_starved(false);

    const int32_t remaining_time = request->get_remaining_time();
    total_exec_time += sequence->estimated_latency();
    if (request->tpot_slo_ms() < min_tpot) {
      min_tpot = request->tpot_slo_ms();
    }
    if (remaining_time < sequence->estimated_latency() + constant_overhead) {
      continue;
    }
    if (remaining_time < min_remaining_time) {
      min_remaining_time = remaining_time;
    }
  }

  if (!for_prefill) {
    int32_t latency_budget_threshold = static_cast<int32_t>(0.65 * min_tpot);
    latency_budget = std::max(min_remaining_time, latency_budget_threshold);
  }

  const double lambda = SchedulerConfig::get_instance().aggressive_coeff();
  double load_judge_func = 0.0;
  if (for_prefill) {
    load_judge_func = total_exec_time + constant_overhead;
  } else {
    const double denominator =
        std::max(latency_budget - constant_overhead, 1e-6);
    load_judge_func = total_exec_time * latency_budget / denominator;
  }

  for (auto it = request_priority_queue->begin();
       it != request_priority_queue->end();
       ++it) {
    auto request = *it;
    auto& sequence = request->sequences()[0];

    if (SchedulerConfig::get_instance().enable_starve_prevent()) {
      const int32_t starve_unit_time = sequence->is_prefill_stage()
                                           ? -request->ttft_slo_ms()
                                           : -request->tpot_slo_ms();
      const int32_t starve_time_threshold = static_cast<int32_t>(
          SchedulerConfig::get_instance().starve_threshold() *
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

  auto comparator = create_comparator("urgency_density", true);
  CHECK(request_priority_queue->supports_sort())
      << "urgency_density requires sortable request queue.";
  request_priority_queue->sort(comparator);

  if (for_prefill && !request_priority_queue->empty()) {
    constexpr int32_t kSmallPositiveTimeMs = 2;
    const int32_t top_remaining_time =
        request_priority_queue->top()->get_remaining_time();
    if (min_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
      latency_budget = min_remaining_time;
    } else if (top_remaining_time > constant_overhead + kSmallPositiveTimeMs) {
      latency_budget = top_remaining_time;
    } else {
      const auto& top_sequence = request_priority_queue->top()->sequences()[0];
      latency_budget = top_sequence->estimated_latency() + constant_overhead +
                       kSmallPositiveTimeMs;
    }
  }
}

void ContinuousScheduler::create_waiting_queue(const Options& options) {
  if (options.priority_strategy() == "urgency_density") {
    // currently only for multi-priority scheduling
    waiting_priority_queue_ = std::make_unique<DequeQueue>();
    waiting_priority_queue_offline_ = std::make_unique<DequeQueue>();
  } else {
    // use default HeapQueue
    waiting_priority_queue_ = std::make_unique<HeapQueue>(
        create_comparator(options.priority_strategy(), false));
    waiting_priority_queue_offline_ = std::make_unique<HeapQueue>(
        create_comparator(options.priority_strategy(), false));
  }
}

void ContinuousScheduler::create_running_queue(const Options& options) {
  if (options.priority_strategy() == "fcfs") {
    // for FCFS, we can use simple deque-based queue since no need to sort
    running_queue_offline_ = std::make_unique<DequeQueue>();
    running_queue_ = std::make_unique<DequeQueue>();
  } else {
    if (options.priority_strategy() == "deadline" ||
        options.priority_strategy() == "priority") {
      running_queue_ = std::make_unique<SetQueue>(
          create_comparator(options.priority_strategy(), true));
      running_queue_offline_ = std::make_unique<SetQueue>(
          create_comparator(options.priority_strategy(), true));
    } else {
      // for other strategies like multi-priority scheduling, we can use
      // deque-based queue and sort them in place when needed
      running_queue_offline_ = std::make_unique<DequeQueue>();
      running_queue_ = std::make_unique<DequeQueue>();
    }
  }
}

bool ContinuousScheduler::check_if_enough_to_evict(
    RequestPriorityQueue* running_queue_to_evict,
    Sequence* prefill_sequence,
    size_t max_handle_num_tokens,
    size_t& num_request_to_evict) {
  // check if it's enough when we evict this requests queue
  auto block_size = kv_cache_manager_->block_size();
  const size_t num_blocks_needed =
      (max_handle_num_tokens + block_size - 1) / block_size;
  size_t num_blocks_can_evict = 0;
  // count the number of blocks can be preempted
  for (auto it = running_queue_to_evict->rbegin();
       it != running_queue_to_evict->rend();
       ++it) {
    std::shared_ptr<Request> request_to_preempt = *it;
    num_request_to_evict++;
    // count the number of blocks belong to the request
    for (const auto& seq : request_to_preempt->sequences()) {
      // num_blocks_can_evict += seq->kv_state().num_blocks(BlockType::KV);
      size_t shared_kv_blocks_num =
          seq->kv_state().shared_blocks_num(BlockType::KV);
      size_t num_kv_blocks = seq->kv_state().num_blocks(BlockType::KV);
      CHECK_GE(num_kv_blocks, shared_kv_blocks_num);
      for (size_t i = 0; i < shared_kv_blocks_num; i++) {
        // if ==2, prefix cache block will be evicted when allocate
        const auto& block = seq->kv_state().blocks(BlockType::KV)[i];
        if (block.ref_count() <= 2) {
          num_blocks_can_evict += 1;
        }
      }
      num_blocks_can_evict += (num_kv_blocks - shared_kv_blocks_num);
    }
    if (num_blocks_needed <= num_blocks_can_evict) {
      return true;
    }
  }
  return false;
}

void ContinuousScheduler::cache_in_batch_prefix(
    const std::vector<Sequence*>& sequences,
    const std::vector<size_t>& current_step_token_budgets) {
  if (!enable_prefix_cache_ || !enable_in_batch_prefix_cache_ ||
      sequences.empty()) {
    return;
  }
  CHECK_EQ(sequences.size(), current_step_token_budgets.size());
  for (size_t i = 0; i < sequences.size(); ++i) {
    Sequence* sequence = sequences[i];
    if (sequence == nullptr || !sequence->is_prefill_stage()) {
      continue;
    }
    const size_t max_handle_num_tokens =
        sequence->kv_state().kv_cache_tokens_num() +
        current_step_token_budgets[i];
    kv_cache_manager_->cache(sequence, max_handle_num_tokens);
  }
}

void ContinuousScheduler::clear_mtp_bootstrap(Request* request) {
  if (!options_.enable_disagg_pd() || options_.num_speculative_tokens() <= 0 ||
      request == nullptr || request->sequences().empty()) {
    return;
  }
  Sequence* sequence = request->sequences()[0].get();
  if (sequence == nullptr) {
    return;
  }
  sequence->clear_mtp_bootstrap_embedding();
}

void ContinuousScheduler::handle_prefill_requests(
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
  if (options_.priority_strategy() == "urgency_density") {
    get_latency_budget_and_request_order(
        waiting_priority_queue, latency_budget, true);
  }

  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  while (!waiting_priority_queue->empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 && latency_budget > estimate_latency) {
    if (!options_.enable_disagg_pd() &&
        kv_cache_manager_->kv_cache_utilization() >=
            ::xllm::SchedulerConfig::get_instance()
                .prefill_scheduling_memory_usage_threshold()) {
      blocks_exhausted = true;
      break;
    }

    std::shared_ptr<Request> request(waiting_priority_queue->top());
    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get());
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the request priority queue
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

      // FIXME: use actual num_tokens to handle
      // Currently overestimating the number of tokens actually processed when
      // enable prefix cache
      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      const int32_t worker_cp_size =
          Platform::uses_model_cp_partition() ? 1 : options_.cp_size();
      num_tokens = maybe_align_cp_prefill_tokens(
          prefill_sequence.get(), num_tokens, worker_cp_size);
      const size_t target_num_tokens =
          prefill_sequence->kv_cache_tokens_num() + num_tokens;
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      // preempt offline decode
      if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                       target_num_tokens)) {
        can_schedule = false;
        if (options_.enable_online_preempt_offline() && !request->offline() &&
            !running_queue_offline_->empty()) {
          size_t num_request_to_evict = 0;
          // according to the prefill_sequence num tokens to check if can
          // allocate blocks for it through evict

          bool enough_to_evict =
              check_if_enough_to_evict(running_queue_offline_.get(),
                                       prefill_sequence.get(),
                                       target_num_tokens,
                                       num_request_to_evict);
          if (enough_to_evict) {
            for (size_t i = 0; i < num_request_to_evict; ++i) {
              std::shared_ptr<Request> request_to_preempt =
                  running_queue_offline_->back();
              ++num_online_prefill_preempt_offline_requests;
              clear_mtp_bootstrap(request_to_preempt.get());
              kv_cache_manager_->deallocate(request_to_preempt.get());
              running_queue_offline_->pop_back();
              // add preemptable request to request priority queue
              // TO IMPROVE?: not process this offline request in current batch
              request_to_preempt->set_preempted();
              waiting_priority_queue_offline_->push(request_to_preempt);
            }
            if (!kv_cache_manager_->allocate(prefill_sequence.get(),
                                             target_num_tokens)) {
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
        if ((estimate_latency + allocated_estimate_latency +
                 seq_estimate_latency >
             latency_budget) &&
            (!running_sequences_.empty())) {
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
    clear_mtp_bootstrap(request.get());
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
      LOG(INFO) << "latency budegt: " << latency_budget
                << ", estimate latency: " << estimate_latency;
      LOG(FATAL) << "Unexpected error: blocks and budget are enough but can "
                    "not schedule.";
    }
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

void ContinuousScheduler::handle_decode_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_offline_decode_preempt_offline_requests,
    size_t& num_online_decode_preempt_online_requests,
    size_t& num_online_decode_preempt_offline_requests,
    RequestPriorityQueue* running_queue) {
  if (options_.priority_strategy() == "urgency_density" &&
      running_queue != nullptr && !running_queue->empty()) {
    get_latency_budget_and_request_order(running_queue, latency_budget, false);
  }

  std::vector<Sequence*> candidate_sequences;
  std::vector<size_t> candidate_token_budgets;

  while (!running_queue->empty() &&
         remaining_token_budget > min_speculative_tokens_required_ &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = running_queue->top();
    // TODO: check if request is timeout

    const size_t num_sequences = request->sequences().size();
    candidate_sequences.clear();
    candidate_token_budgets.clear();
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;

    if (request->check_beam_search()) {
      std::vector<Sequence*> active_sequences;
      active_sequences.reserve(num_sequences);
      for (auto& seq : request->sequences()) {
        if (!seq->finished()) {
          active_sequences.emplace_back(seq.get());
        }
      }
      if (active_sequences.empty()) {
        running_queue->pop_top();
        continue;
      }

      const size_t decode_step_tokens = min_speculative_tokens_required_ + 1;
      if (decode_step_tokens * active_sequences.size() >
              remaining_token_budget ||
          active_sequences.size() > remaining_seq_budget) {
        has_enough_budget = false;
      }

      if (has_enough_budget && options_.enable_latency_aware_schedule() &&
          !(options_.instance_role().has_value() &&
            options_.instance_role().value() == InstanceRole::PREFILL)) {
        for (auto* sequence : active_sequences) {
          const double seq_estimate_latency =
              profile_manager_->predict_step_time(sequence, false);
          if (estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              latency_budget) {
            has_enough_budget = false;
            break;
          }
          allocated_estimate_latency += seq_estimate_latency;
        }
      }

      // Reset estimation value. It will be recomputed on successful allocation.
      allocated_estimate_latency = 0.0;

      if (has_enough_budget) {
        const size_t block_size = kv_cache_manager_->block_size();
        size_t needed_blocks = 0;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + min_speculative_tokens_required_;
          needed_blocks += estimate_decode_extra_blocks(
              sequence, updated_num_tokens, block_size);
        }

        const int32_t dp_rank = active_sequences.front()->dp_rank();
        const size_t free_blocks =
            get_sequence_free_blocks_for_rank(kv_cache_manager_, dp_rank);
        if (needed_blocks > free_blocks) {
          has_enough_blocks = false;
        }
      }

      if (has_enough_budget && has_enough_blocks) {
        bool allocate_failed = false;
        for (auto* sequence : active_sequences) {
          const size_t updated_num_tokens =
              sequence->num_tokens() + min_speculative_tokens_required_;
          if (!kv_cache_manager_->allocate(sequence, updated_num_tokens)) {
            allocate_failed = true;
            break;
          }
          if (sequence->if_cache_block_for_prefill()) {
            kv_cache_manager_->cache(sequence);
          }
          candidate_sequences.emplace_back(sequence);
          candidate_token_budgets.emplace_back(decode_step_tokens);
          allocated_tokens += decode_step_tokens;
          allocated_seqs += 1;
        }

        if (allocate_failed) {
          LOG(ERROR) << "Beam strict scheduling allocation failed. "
                     << "request_id=" << request->request_id()
                     << ", beam=" << request->check_beam_search();
          // Fallback to full request deallocation to avoid inconsistent
          // per-sequence states.
          clear_mtp_bootstrap(request.get());
          kv_cache_manager_->deallocate(request.get());
          running_queue->pop_top();
          request->set_preempted();
          if (request->offline()) {
            waiting_priority_queue_offline_->push(request);
          } else {
            waiting_priority_queue_->push(request);
          }
          continue;
        }

        if (options_.enable_latency_aware_schedule() &&
            !(options_.instance_role().has_value() &&
              options_.instance_role().value() == InstanceRole::PREFILL)) {
          for (auto* sequence : candidate_sequences) {
            allocated_estimate_latency +=
                profile_manager_->predict_step_time(sequence, false);
          }
        }
      }
    } else {
      for (auto& sequence : request->sequences()) {
        if (sequence->finished()) {
          continue;
        }
        // no budget left
        double seq_estimate_latency = 0;
        if (options_.enable_latency_aware_schedule()
            // force not enabled on prefill node (only offline req decode here)
            && !(options_.instance_role().has_value() &&
                 options_.instance_role().value() == InstanceRole::PREFILL)) {
          seq_estimate_latency =
              profile_manager_->predict_step_time(sequence.get(), false);
          if (estimate_latency + allocated_estimate_latency +
                  seq_estimate_latency >
              latency_budget) {
            has_enough_budget = false;
            break;
          }
        }
        if (allocated_tokens + min_speculative_tokens_required_ >=
                remaining_token_budget ||
            allocated_seqs >= remaining_seq_budget) {
          has_enough_budget = false;
          break;
        }
        // sequence token already appended
        size_t updated_num_tokens =
            sequence->num_tokens() + min_speculative_tokens_required_;
        // no blocks left
        if (!kv_cache_manager_->allocate(sequence.get(), updated_num_tokens)) {
          has_enough_blocks = false;
          break;
        }

        if (sequence->if_cache_block_for_prefill()) {
          kv_cache_manager_->cache(sequence.get());
        }

        // update the allocated tokens for the sequence
        allocated_tokens += min_speculative_tokens_required_ + 1;
        allocated_seqs += 1;
        allocated_estimate_latency += seq_estimate_latency;
        candidate_sequences.emplace_back(sequence.get());
        candidate_token_budgets.emplace_back(min_speculative_tokens_required_ +
                                             1);
      }
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    // schedule candidates in the request if there are enough blocks
    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue->pop_top();
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
      estimate_latency += allocated_estimate_latency;

      continue;
    }

    // budget exhausted, do partially schedule the request
    if (!has_enough_budget) {
      handle_abnormal_request(running_queue,
                              candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              allocated_estimate_latency,
                              remaining_token_budget,
                              remaining_seq_budget,
                              estimate_latency,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      break;
    }

    // memory exhausted, try to preempt lowest priority request
    // continue to evict blocks until enough or no other requests that can be
    // preempted
    if (options_.enable_online_preempt_offline() && !request->offline() &&
        !running_queue_offline_->empty()) {
      std::shared_ptr<Request> request_to_preempt =
          running_queue_offline_->back();
      ++num_online_decode_preempt_offline_requests;
      clear_mtp_bootstrap(request_to_preempt.get());
      kv_cache_manager_->deallocate(request_to_preempt.get());
      running_queue_offline_->pop_back();
      // add preemptable request to waiting priority queue
      request_to_preempt->set_preempted();
      waiting_priority_queue_offline_->push(request_to_preempt);
      continue;
    } else if (running_queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue->back();
      if (request_to_preempt.get() != request.get()) {
        // TO IMPROVE: kv cache offload to cpu
        clear_mtp_bootstrap(request_to_preempt.get());
        kv_cache_manager_->deallocate(request_to_preempt.get());
        running_queue->pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        if (request_to_preempt->offline()) {
          ++num_offline_decode_preempt_offline_requests;
          waiting_priority_queue_offline_->push(request_to_preempt);
        } else {
          ++num_online_decode_preempt_online_requests;
          waiting_priority_queue_->push(request_to_preempt);
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(running_queue,
                            candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            allocated_estimate_latency,
                            remaining_token_budget,
                            remaining_seq_budget,
                            estimate_latency,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    break;
  }
}

// NOTE: refactor ChunkedPrefillScheduler and ContinuousScheduler later.
void ContinuousScheduler::handle_abnormal_request(
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
    bool blocks_exhausted) {
  std::shared_ptr<Request> request = running_queue->top();
  if (candidate_sequences.empty()) {
    if (!running_sequences_.empty()) {
      return;
    }

    // unknown case, maybe a schdule bug.
    if (budget_exhausted && blocks_exhausted) {
      LOG(FATAL) << "Unknown case, budget and blocks are not exhausted, but "
                    "there are no running sequences."
                 << " budget_exhausted = " << budget_exhausted
                 << " blocks_exhausted = " << blocks_exhausted
                 << " candidate_sequences.size = " << candidate_sequences.size()
                 << ", running_sequences.size = " << running_sequences_.size();
    }

    // budget exhausted
    if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, please set a larger "
                    "max_tokens value via --max_tokens_per_batch.";
    } else {
      CHECK(running_queue->size() == 1)
          << "Running queue size is not 1, there maybe a bug of request "
             "preemption logic. running_queue_.size ="
          << running_queue->size();
      if (util::sum(kv_cache_manager_->num_used_blocks()) !=
          request->total_num_blocks()) {
        // blocks_exhausted is true.
        // NOTE: consider dp > 1, here we need get all num blocks in use.
        // Total num blocks in use not equal request->total_num_blocks() means
        // some sequences are not scheduled but hold blocks in disagg PD mode.
        return;
      }
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
    }

    // request is too long, budget or memory no enough.
    running_queue->pop_top();
    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough resource to schedule a single sequence"});
  } else {
    // partially schedule the sequences in request
    if (!request->check_beam_search()) {
      running_queue->pop_top();
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      estimate_latency += allocated_estimate_latency;
    }
  }
}

void ContinuousScheduler::handle_running_requests(
    std::shared_ptr<Request> request) {
  if (request->finished() || request->cancelled()) {
    LOG(FATAL) << "Unknow error, finished/cancelled request have be handled "
                  "before. request_id is "
               << request->request_id();
  }

  // check if the request can be expanded
  if (request->expand_sequences()) {
    // cache the blocks to share among the sequences
    kv_cache_manager_->cache(request->sequences()[0].get());
  }

  // release blocks for finished sequences here
  for (auto& sequence : request->sequences()) {
    if (sequence->finished()) {
      kv_cache_manager_->deallocate(sequence.get());
    }
  }
}

std::vector<Batch> ContinuousScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  // Include those requests that are preempted by others.
  std::shared_ptr<Request> request;
  // read from request queue then push to request priority queue
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
      clear_mtp_bootstrap(request.get());
      kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

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
        if ((*it)->offline()) {
          running_queue_offline_->push(*it, last_step_prefill_);
        } else {
          running_queue_->push(*it, last_step_prefill_);
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
    for (auto it = running_requests_.cbegin(); it != running_requests_.cend();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      if ((*it)->offline()) {
        running_queue_offline_->push(*it);
      } else {
        running_queue_->push(*it);
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
  double estimate_latency = profile_manager_->get_constant_overhead();
  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;
  size_t num_offline_decode_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_online_requests = 0;
  size_t num_online_prefill_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_offline_requests = 0;
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

  if (running_sequences_.empty()) {
    latency_budget = options_.max_global_tpot_ms();
    // Handle decoding requests.
    // no prefill request, schedule the decode requests in the running priority
    // queue
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

std::vector<Batch> ContinuousScheduler::schedule_request(
    const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  std::vector<Batch> batch;
  while (true) {
    apply_cancel_requests();
    batch = prepare_batch();
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (!all_empty) {
      return batch;
    }

    if (if_queue_not_empty()) {
      continue;
    }

    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 1;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }
  // return an empty batch
  return batch;
}

void ContinuousScheduler::apply_cancel_requests() {
  std::vector<std::shared_ptr<Request>> requests =
      cancel_request_queue_->take_all();
  for (const std::shared_ptr<Request>& request : requests) {
    request->set_cancel();
  }
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousScheduler::step(const absl::Duration& timeout) {
  if (try_complete_pause()) {
    return;
  }

  // Check if paused - block instead of busy-waiting.
  //
  // step() is called in a tight loop by LLMMaster::run() with no sleep, so a
  // bare `return` here would spin a CPU core at 100% while paused. Block on
  // pause_cv_ until resume() flips the state. resume() holds pause_mutex_ when
  // storing RUNNING and then notifies, so there is no lost-wakeup window.
  //
  // We use wait_for with a bounded timeout rather than an unbounded wait: the
  // owning LLMMaster signals shutdown via its own `stoped_` flag (not visible
  // here) and joins this loop thread WITHOUT calling resume() (see
  // ~LLMMaster). An unbounded wait would therefore deadlock shutdown if the
  // engine is destroyed while paused. The timeout lets the loop periodically
  // fall through so the `stoped_` check in LLMMaster::run() can break the loop.
  if (pause_state_.load(std::memory_order_acquire) == PauseState::PAUSED) {
    std::unique_lock<std::mutex> lock(pause_mutex_);
    pause_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
      return pause_state_.load(std::memory_order_acquire) != PauseState::PAUSED;
    });
    return;  // Stay paused (or fall through to shutdown check on next loop)
  }

  if (!options_.enable_schedule_overlap()) {
    // get a new batch of requests
    last_batch_lengths_.clear();
    std::vector<Batch> batch = schedule_request(timeout);
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (all_empty) {
      return;
    }

    if (!options_.enable_pd_ooc()) {
      engine_->step(batch);
    } else {
      step_with_pd_ooc(batch);
    }

    // process request output in batch
    process_batch_output(false);
  } else {
    step_with_schedule_overlap(timeout);
  }
}

void ContinuousScheduler::step_with_schedule_overlap(
    const absl::Duration& timeout) {
  // get a new batch of requests
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_all_empty =
      std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      });
  bool last_batch_all_empty = std::all_of(
      last_batch_.begin(), last_batch_.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      });
  if (cur_batch_all_empty && last_batch_all_empty) {
    return;
  }

  if (!cur_batch_all_empty) {
    engine_->step(batch);
  }

  // producer-consumer mode, make sure only one step is scheduled in advance
  if (!is_first_step_ && !last_batch_all_empty) {
    engine_->update_last_step_result(last_batch_);
    process_batch_output(true);
  }
  last_batch_ = std::move(batch);
  last_running_sequences_ = running_sequences_;
  last_running_requests_ = running_requests_;
  is_first_step_ = false;
}

void ContinuousScheduler::generate() {
  bool batch_empty = false;
  while (num_pending_requests() > 0 || !batch_empty ||
         request_queue_.size() > 0) {
    // build a batch of requests/sequences
    const auto timeout = absl::Milliseconds(50);
    std::vector<Batch> batch = schedule_request(timeout);
    batch_empty = true;
    for (auto& b : batch) {
      batch_empty &= b.empty();
    }
    if (batch_empty) {
      continue;
    }

    // run inference for the batch
    engine_->step(batch);

    // process request output in batch
    process_batch_output(false);
  }

  // wait for all responses done
  response_processor_->wait_completion();
}

int64_t ContinuousScheduler::amortized_token_latency_ms(int64_t tbt_ms,
                                                        size_t num_tokens) {
  const int64_t n = static_cast<int64_t>(num_tokens);
  return (tbt_ms + n / 2) / n;
}

void ContinuousScheduler::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  const auto now = absl::Now();
  const bool speculative_metrics_enabled =
      options_.num_speculative_tokens() > 0;
  int64_t step_committed_tokens = 0;
  int64_t step_decode_seqs = 0;
  for (Sequence* sequence : sequences) {
    if (sequence->is_chunked_prefill_stage() ||
        sequence->last_token_handled()) {
      // skip chunked prefill stage
      continue;
    }
    // Read the committed-token count before tbt(), which resets it.
    const size_t committed_tokens = sequence->generated_tokens_since_latency();
    int64_t tbt_milliseconds = sequence->tbt(now);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
    } else {
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
      if (speculative_metrics_enabled && committed_tokens > 0) {
        HISTOGRAM_OBSERVE(
            speculative_per_token_latency_milliseconds,
            amortized_token_latency_ms(tbt_milliseconds, committed_tokens));
        step_committed_tokens += static_cast<int64_t>(committed_tokens);
        ++step_decode_seqs;
      }
    }
  }
  if (step_decode_seqs > 0) {
    GAUGE_SET(speculative_mean_tokens_per_decode_step,
              static_cast<double>(step_committed_tokens) / step_decode_seqs);
  }
}

void ContinuousScheduler::process_batch_output(bool enable_schedule_overlap) {
  std::vector<Sequence*>& to_be_processed_sequences =
      enable_schedule_overlap ? last_running_sequences_ : running_sequences_;
  std::vector<std::shared_ptr<Request>>& to_be_processed_requests =
      enable_schedule_overlap ? last_running_requests_ : running_requests_;
  // Beam search may replace Sequence objects inside SequencesGroup.
  // Always refresh the sequence pointers from requests before dereferencing.
  refresh_sequences_from_requests(to_be_processed_requests,
                                  to_be_processed_sequences);
  // update token latency metrics
  update_token_latency_metrics(to_be_processed_sequences);

  // update slot usage and activation metrics
  update_memory_metrics(to_be_processed_sequences);

  std::vector<std::shared_ptr<Request>> stream_requests;
  // process request output in batch
  for (auto request : to_be_processed_requests) {
    // ignore cancelled/finished requests when enable_schedule_overlap.
    if (options_.enable_schedule_overlap()) {
      if (request->state().stream) {
        if (request->cancelled()) {
          continue;
        }
        if (!request->finished()) {
          stream_requests.emplace_back(request);
          continue;
        }
        // handle token when last token not be handled.
        if (request->finished() && !request->last_token_handled()) {
          request->handle_last_token();
          stream_requests.emplace_back(request);
        }
      } else if (request->finished() && !request->last_token_handled()) {
        request->handle_last_token();
      }
    } else if (request->state().stream) {
      stream_requests.emplace_back(request);
    }
  }
  if (!stream_requests.empty()) {
    response_processor_->process_stream_requests(stream_requests);
  }
}

void ContinuousScheduler::refresh_sequences_from_requests(
    const std::vector<std::shared_ptr<Request>>& requests,
    std::vector<Sequence*>& sequences) const {
  sequences.clear();
  for (const auto& request : requests) {
    if (request == nullptr) {
      continue;
    }
    auto& request_sequences = request->sequences();
    for (auto& sequence : request_sequences) {
      if (sequence != nullptr) {
        sequences.emplace_back(sequence.get());
      }
    }
  }
}

std::vector<int64_t> ContinuousScheduler::get_num_occupied_slots(
    std::vector<Sequence*>& sequences) const {
  std::vector<int64_t> num_occupied_slots(options_.dp_size());
  std::vector<int64_t> num_unfilled_blocks(options_.dp_size());
  std::vector<size_t> num_used_blocks = kv_cache_manager_->num_used_blocks();

  auto block_size = kv_cache_manager_->block_size();

  for (auto& sequence : sequences) {
    const int32_t dp_rank = sequence->dp_rank();
    // last_block_len is the length of the last unfilled block of each
    // sequence.
    int32_t last_block_len =
        sequence->kv_state().kv_cache_tokens_num() % block_size;
    num_occupied_slots[dp_rank] += last_block_len;
    num_unfilled_blocks[dp_rank] += last_block_len > 0 ? 1 : 0;
  }

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    num_occupied_slots[dp_rank] +=
        (num_used_blocks[dp_rank] - num_unfilled_blocks[dp_rank]) * block_size;
  }
  return num_occupied_slots;
}

std::vector<int64_t> ContinuousScheduler::get_active_activation_in_bytes() {
  std::vector<int64_t> all_active_activation_in_bytes =
      engine_->get_active_activation_memory();
  std::vector<int64_t> active_activation_in_bytes(options_.dp_size());

  const int32_t dp_local_tp_size =
      all_active_activation_in_bytes.size() / options_.dp_size();

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    active_activation_in_bytes[dp_rank] =
        all_active_activation_in_bytes[dp_rank * dp_local_tp_size];
  }
  return active_activation_in_bytes;
}

void ContinuousScheduler::update_memory_metrics(
    std::vector<Sequence*>& sequences) {
  if (sequences.empty()) {
    return;
  }
  std::vector<int64_t> num_occupied_slots = get_num_occupied_slots(sequences);
  std::vector<int64_t> active_activation_size_in_bytes =
      get_active_activation_in_bytes();
  int64_t num_total_slots =
      kv_cache_manager_->num_blocks() * kv_cache_manager_->block_size();

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    double occupied_slots_ratio =
        static_cast<double>(num_occupied_slots[dp_rank]) / num_total_slots;
    double active_kv_cache_size_in_kilobytes =
        occupied_slots_ratio * GAUGE_VALUE(total_kv_cache_size_in_kilobytes);
    int64_t active_activation_size_in_kilobytes =
        active_activation_size_in_bytes[dp_rank] / 1024;

    MULTI_HISTOGRAM_OBSERVE(
        active_kv_cache_size_in_kilobytes,
        std::to_string(dp_rank),
        static_cast<int64_t>(active_kv_cache_size_in_kilobytes));

    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      MULTI_HISTOGRAM_OBSERVE(decode_active_activation_size_in_kilobytes,
                              std::to_string(dp_rank),
                              active_activation_size_in_kilobytes);
    } else {
      if (sequences[0]->is_first_token()) {
        MULTI_HISTOGRAM_OBSERVE(prefill_active_activation_size_in_kilobytes,
                                std::to_string(dp_rank),
                                active_activation_size_in_kilobytes);
      } else {
        MULTI_HISTOGRAM_OBSERVE(decode_active_activation_size_in_kilobytes,
                                std::to_string(dp_rank),
                                active_activation_size_in_kilobytes);
      }
    }
  }
}

void ContinuousScheduler::step_with_pd_ooc(std::vector<Batch>& batch) {
  for (size_t i = 0; i < batch.size(); i++) {
    for (size_t j = 0; j < batch[i].size(); j++) {
      last_batch_lengths_.push_back(batch[i][j]->num_tokens());
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  engine_->step(batch);
  auto end = std::chrono::high_resolution_clock::now();
  double duration_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;

  std::stringstream ss;
  ss << "bs=" << last_batch_lengths_.size() << " - [";
  for (size_t i = 0; i < last_batch_lengths_.size(); ++i) {
    ss << last_batch_lengths_[i];
    if (i != last_batch_lengths_.size() - 1) ss << ", ";
  }
  ss << "]";
  VLOG(1) << "PERF - " << ss.str() << " - " << std::fixed
          << std::setprecision(3) << duration_ms << " ms";
}

bool ContinuousScheduler::try_complete_pause() {
  if (pause_state_.load(std::memory_order_acquire) != PauseState::PAUSING) {
    return false;
  }

  const PauseMode mode = pause_mode_.load(std::memory_order_acquire);

  // WAIT mode: do not preempt. Let already-running requests finish naturally,
  // and only transition to PAUSED once nothing is left running/in-flight. We
  // return false here so step() proceeds with normal scheduling to drain them.
  // (Matches vLLM "wait": ongoing requests complete before the engine pauses.)
  if (mode == PauseMode::WAIT) {
    const bool last_batch_in_flight =
        options_.enable_schedule_overlap() && !is_first_step_ &&
        !std::all_of(last_batch_.begin(),
                     last_batch_.end(),
                     [](const Batch& b) { return b.empty(); });
    if (!running_requests_.empty() || last_batch_in_flight) {
      return false;  // still draining; keep stepping normally
    }
    {
      std::lock_guard<std::mutex> lock(pause_mutex_);
      pause_state_.store(PauseState::PAUSED, std::memory_order_release);
    }
    pause_cv_.notify_all();
    LOG(INFO) << "Scheduler paused (WAIT mode: all in-flight requests drained, "
                 "KV cache preserved)";
    return true;
  }

  // KEEP / ABORT: drain the in-flight overlap pipeline first.
  //
  // With enable_schedule_overlap, a forward batch may still be in flight on the
  // device (tracked by last_batch_). We must collect its results and let the
  // sequence/KV state settle BEFORE deallocating KV cache, otherwise we would
  // free blocks that the in-flight forward still reads/writes, corrupting the
  // recomputation after resume (garbled output).
  if (options_.enable_schedule_overlap() && !is_first_step_) {
    const bool last_batch_all_empty = std::all_of(
        last_batch_.begin(), last_batch_.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (!last_batch_all_empty) {
      // Drain the one in-flight step scheduled in advance.
      engine_->update_last_step_result(last_batch_);
      process_batch_output(true);
    }
    // Reset overlap pipeline bookkeeping so resume starts clean.
    last_batch_.clear();
    last_batch_.resize(options_.dp_size());
    last_running_requests_.clear();
    last_running_sequences_.clear();
    is_first_step_ = true;
  }

  // Now the pipeline is drained; handle running requests per mode.
  if (mode == PauseMode::ABORT) {
    abort_all_running_requests();
    {
      std::lock_guard<std::mutex> lock(pause_mutex_);
      pause_state_.store(PauseState::PAUSED, std::memory_order_release);
    }
    pause_cv_.notify_all();
    LOG(INFO)
        << "Scheduler paused (ABORT mode: all running requests cancelled)";
  } else {  // KEEP
    preempt_all_running_requests();
    {
      std::lock_guard<std::mutex> lock(pause_mutex_);
      pause_state_.store(PauseState::PAUSED, std::memory_order_release);
    }
    pause_cv_.notify_all();
    LOG(INFO) << "Scheduler paused (KEEP mode: requests preempted to waiting "
                 "queue, KV cache freed, will re-prefill on resume)";
  }
  return true;
}

// ============== Async RL training support: Pause/Resume ==============
void ContinuousScheduler::reset_prefix_cache() {
  if (kv_cache_manager_ != nullptr) {
    kv_cache_manager_->reset_prefix_cache();
  }
}

void ContinuousScheduler::pause(PauseMode mode) {
  const char* mode_str = mode == PauseMode::KEEP    ? "KEEP"
                         : mode == PauseMode::ABORT ? "ABORT"
                                                    : "WAIT";
  LOG(INFO) << "Pausing scheduler (mode=" << mode_str << ")";

  // Publish the mode before the state so the loop thread, upon observing
  // PAUSING, reads a consistent mode.
  pause_mode_.store(mode, std::memory_order_relaxed);

  PauseState expected = PauseState::RUNNING;
  if (!pause_state_.compare_exchange_strong(expected,
                                            PauseState::PAUSING,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
    LOG(WARNING) << "Scheduler already paused or pausing";
    return;
  }

  LOG(INFO) << "Scheduler pause requested (mode=" << mode_str
            << "). Running requests: " << running_requests_.size();
}

bool ContinuousScheduler::wait_until_paused(int64_t timeout_ms) {
  std::unique_lock<std::mutex> lock(pause_mutex_);
  // Wait until the transition settles: either fully PAUSED, or no longer
  // pausing at all (e.g. a concurrent resume() moved it back to RUNNING).
  // This avoids hanging forever if pause() was a no-op or resume() raced in.
  auto settled = [this] {
    return pause_state_.load(std::memory_order_acquire) != PauseState::PAUSING;
  };
  bool ok;
  if (timeout_ms < 0) {
    pause_cv_.wait(lock, settled);
    ok = true;
  } else {
    ok = pause_cv_.wait_for(
        lock, std::chrono::milliseconds(timeout_ms), settled);
  }
  // Only report "paused" if we actually ended up PAUSED.
  return ok &&
         pause_state_.load(std::memory_order_acquire) == PauseState::PAUSED;
}

void ContinuousScheduler::resume() {
  LOG(INFO) << "Resuming scheduler";

  // Resume from either PAUSING or PAUSED. Using exchange() unconditionally sets
  // RUNNING and returns the previous state, so a resume() issued before step()
  // has advanced PAUSING -> PAUSED still works. Hold the lock and notify so any
  // thread blocked in wait_until_paused() is released.
  PauseState prev;
  {
    std::lock_guard<std::mutex> lock(pause_mutex_);
    prev =
        pause_state_.exchange(PauseState::RUNNING, std::memory_order_acq_rel);
  }
  pause_cv_.notify_all();
  if (prev == PauseState::RUNNING) {
    LOG(WARNING) << "Scheduler was not paused; resume() is a no-op";
    return;
  }

  LOG(INFO) << "Scheduler resumed. Preempted requests in waiting queue: "
            << get_waiting_requests_num()
            << " (will need re-prefill with new weights)";
}

bool ContinuousScheduler::is_paused() const {
  auto state = pause_state_.load(std::memory_order_acquire);
  return state == PauseState::PAUSED || state == PauseState::PAUSING;
}

void ContinuousScheduler::preempt_all_running_requests() {
  const size_t total_to_preempt = running_requests_.size() +
                                  running_queue_->size() +
                                  running_queue_offline_->size();
  if (total_to_preempt == 0) {
    return;
  }

  LOG(INFO) << "Preempting " << total_to_preempt
            << " running requests for pause";

  size_t preempted_count = 0;

  // Preempt a single request: free its KV cache and move it back to the
  // matching waiting queue so it will be re-prefilled on resume.
  auto preempt_one = [&](const std::shared_ptr<Request>& request) {
    if (!request) {
      return;
    }

    // Skip already finished requests
    if (request->finished()) {
      return;
    }

    // Deallocate KV cache blocks (critical for RL weight updates)
    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());

    // Mark as preempted
    request->set_preempted();

    // Push back to waiting queue (will need re-prefill on resume)
    if (request->offline()) {
      waiting_priority_queue_offline_->push(request);
    } else {
      waiting_priority_queue_->push(request);
    }

    preempted_count++;
  };

  // 1. Requests selected into the current batch.
  for (auto& request : running_requests_) {
    preempt_one(request);
  }

  // 2. Active decoding requests still waiting in the running queues. These were
  // pushed back to running_queue_/running_queue_offline_ at the start of the
  // step but were not selected into running_requests_ (e.g. budget exhausted,
  // or this step scheduled prefill so decode was skipped). They still hold KV
  // cache and must be preempted as well.
  while (!running_queue_->empty()) {
    preempt_one(running_queue_->top());
    running_queue_->pop_top();
  }
  while (!running_queue_offline_->empty()) {
    preempt_one(running_queue_offline_->top());
    running_queue_offline_->pop_top();
  }

  // Clear running state
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  LOG(INFO) << "Preempted " << preempted_count
            << " requests, KV cache freed, moved to waiting queue";
}

void ContinuousScheduler::abort_all_running_requests() {
  const size_t total_to_abort = running_requests_.size() +
                                running_queue_->size() +
                                running_queue_offline_->size();
  if (total_to_abort == 0) {
    return;
  }

  LOG(INFO) << "Aborting " << total_to_abort << " running requests";

  size_t aborted_count = 0;

  // Abort a single request: free its KV cache and notify the client. Unlike
  // KEEP, aborted requests are NOT pushed back to the waiting queue.
  auto abort_one = [&](const std::shared_ptr<Request>& request) {
    if (!request) {
      return;
    }
    if (request->finished()) {
      return;
    }

    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());
    request->set_cancel();
    response_processor_->process_failed_request(
        request,
        {StatusCode::CANCELLED, "Request aborted due to scheduler pause"});
    aborted_count++;
  };

  // 1. Requests selected into the current batch.
  for (auto& request : running_requests_) {
    abort_one(request);
  }

  // 2. Active decoding requests still waiting in the running queues. These were
  // pushed back to running_queue_/running_queue_offline_ at the start of the
  // step but were not selected into running_requests_ (e.g. budget exhausted,
  // or this step scheduled prefill so decode was skipped). They still hold KV
  // cache and must be aborted as well, otherwise their blocks leak and the
  // client never receives a cancellation.
  while (!running_queue_->empty()) {
    abort_one(running_queue_->top());
    running_queue_->pop_top();
  }
  while (!running_queue_offline_->empty()) {
    abort_one(running_queue_offline_->top());
    running_queue_offline_->pop_top();
  }

  // Clear running state.
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  LOG(INFO) << "Aborted " << aborted_count
            << " requests, KV cache freed (not rescheduled)";
}

}  // namespace xllm

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

#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch_factory.h"
#include "framework/model/model_args.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "scheduler/request_priority_queue.h"
#include "scheduler/scheduler_policy.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

constexpr absl::Duration kDecodeRestoreTimeout = absl::Seconds(60);
constexpr char kDecodeRestoreTimeoutMessage[] =
    "Decode request could not reacquire device KV cache within 60 seconds";

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
      batch_mode_(create_batch_mode(options)),
      engine_(engine),
      request_queue_(::xllm::RecConfig::get_instance().request_queue_size()) {
  CHECK(engine_ != nullptr);
  prefetch_admission_limit_ = static_cast<size_t>(
      ::xllm::RecConfig::get_instance().request_queue_size());

  kv_cache_manager_ = engine_->block_manager_pool();
  CHECK(kv_cache_manager_ != nullptr);
  scheduler_metrics_ =
      std::make_unique<SchedulerMetrics>(engine_,
                                         kv_cache_manager_,
                                         options_.dp_size(),
                                         options_.num_speculative_tokens(),
                                         options_.enable_disagg_pd());

  enable_prefix_cache_ =
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
  has_linear_attention_layers_ =
      ::xllm::has_linear_attention_layers(engine_->model_args());
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

  // Construct the scheduling policy from the resolved BatchMode.
  policy_ = create_scheduler_policy(batch_mode_, options_);

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
  create_queues(options);
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

  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    if (request_queue_.size() + prefetch_admission_slots_ >=
        prefetch_admission_limit_) {
      return false;
    }
    ++prefetch_admission_slots_;
  }

  kv_cache_manager_->prefetch_from_storage(request);
  if (kv_cache_manager_->update_prefetch_result(request,
                                                options_.prefetch_timeout())) {
    if (request->finished() || request->cancelled()) {
      release_prefetch_admission_slot();
      VLOG(1) << "[Mooncake][AdmissionCancelled] request="
              << request->request_id();
      return true;
    }
    if (!enqueue_ready_request(request)) {
      std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
      prefetch_admission_queue_.emplace_back(request);
      return true;
    }
    release_prefetch_admission_slot();
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    prefetch_admission_queue_.emplace_back(request);
  }
  VLOG(1) << "[Mooncake][AdmissionPending] request=" << request->request_id();
  return true;
}

bool ContinuousScheduler::enqueue_ready_request(
    std::shared_ptr<Request> request) {
  return request_queue_.write(std::move(request));
}

size_t ContinuousScheduler::num_prefetch_pending_requests() const {
  std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
  return prefetch_admission_slots_;
}

void ContinuousScheduler::release_prefetch_admission_slot() {
  std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
  CHECK_GT(prefetch_admission_slots_, 0u);
  --prefetch_admission_slots_;
}

void ContinuousScheduler::drain_prefetched_requests() {
  std::deque<std::shared_ptr<Request>> pending;
  {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    pending.swap(prefetch_admission_queue_);
  }

  std::deque<std::shared_ptr<Request>> still_pending;
  for (std::shared_ptr<Request>& request : pending) {
    if (!kv_cache_manager_->update_prefetch_result(
            request, options_.prefetch_timeout())) {
      still_pending.emplace_back(std::move(request));
      continue;
    }

    if (request->finished() || request->cancelled()) {
      release_prefetch_admission_slot();
      VLOG(1) << "[Mooncake][AdmissionCancelled] request="
              << request->request_id();
      continue;
    }

    if (!enqueue_ready_request(request)) {
      still_pending.emplace_back(std::move(request));
      continue;
    }
    release_prefetch_admission_slot();
    VLOG(1) << "[Mooncake][AdmissionReady] request=" << request->request_id();
  }

  if (!still_pending.empty()) {
    std::lock_guard<std::mutex> lock(prefetch_admission_mutex_);
    prefetch_admission_queue_.insert(
        prefetch_admission_queue_.end(),
        std::make_move_iterator(still_pending.begin()),
        std::make_move_iterator(still_pending.end()));
  }
}

void ContinuousScheduler::create_queues(const Options& options) {
  if (options.priority_strategy() == "multi_slo_and_prio" ||
      options.priority_strategy() == "fcfs") {
    prefill_queue_ = std::make_unique<DequeQueue>();
    chunk_queue_ = std::make_unique<DequeQueue>();
    decode_queue_ = std::make_unique<DequeQueue>();
  } else {
    auto prefill_cmp = create_comparator(options.priority_strategy(), false);
    auto decode_cmp = create_comparator(options.priority_strategy(), true);
    prefill_queue_ = std::make_unique<HeapQueue>(prefill_cmp);
    chunk_queue_ = std::make_unique<SetQueue>(decode_cmp);
    decode_queue_ = std::make_unique<SetQueue>(decode_cmp);
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

void ContinuousScheduler::drain_decode_restore_waiting(
    std::vector<std::shared_ptr<Request>>& finished) {
  const absl::Time now = absl::Now();
  for (auto it = decode_restore_waiting_.begin();
       it != decode_restore_waiting_.end();) {
    std::shared_ptr<Request>& request = it->request;
    CHECK(request != nullptr);
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      clear_mtp_bootstrap(request.get());
      kv_cache_manager_->deallocate(request.get());
      finished.emplace_back(request);
      it = decode_restore_waiting_.erase(it);
      continue;
    }
    if (now - it->started_at < kDecodeRestoreTimeout) {
      ++it;
      continue;
    }

    clear_mtp_bootstrap(request.get());
    kv_cache_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED, kDecodeRestoreTimeoutMessage});
    it = decode_restore_waiting_.erase(it);
  }
}

std::vector<Batch> ContinuousScheduler::prepare_batch() {
  Timer timer;
  drain_prefetched_requests();
  auto state = make_state();

  // Common phases (strategy-independent)
  policy_->drain_request_queue(state, request_queue_);
  auto finished = policy_->collect_finished(state);
  drain_decode_restore_waiting(finished);

  // Initialize budget
  ScheduleBudget budget;
  budget.estimate_latency = profile_manager_->get_constant_overhead();
  budget.remaining_token_budget = options_.enable_profile_token_budget()
                                      ? profile_manager_->get_token_budget()
                                      : options_.max_tokens_per_batch();
  budget.remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  budget.latency_budget = options_.max_global_tpot_ms();
  budget.num_preempted_requests = 0;
  if (::xllm::SchedulerConfig::get_instance().enable_dp_fair_token_budget() &&
      options_.dp_size() > 1 && options_.instance_role().has_value() &&
      options_.instance_role().value() == InstanceRole::PREFILL) {
    // Fair per-group token budget: each DP group can receive at most
    // max_tokens_per_batch / dp_size tokens per scheduling round, which also
    // bounds the DSV4 SWA burst on any single rank to the per-group share.
    // Anchor the cap to max_tokens_per_batch (not the profile token budget,
    // which may exceed it) so it matches the KV cache estimation burst.
    const int64_t dp_size = options_.dp_size();
    const int64_t max_batch_tokens = options_.max_tokens_per_batch();
    int64_t per_group_cap = (max_batch_tokens + dp_size - 1) / dp_size;
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      // Floor the share at one prefill chunk so a single long sequence
      // still advances at full chunk speed even when the budget is below
      // dp_size * chunk.
      per_group_cap = std::max(
          per_group_cap,
          std::min<int64_t>(options_.max_tokens_per_chunk_for_prefill(),
                            max_batch_tokens));
    } else {
      // Non-chunked prefill computes a whole sequence in one round; a share
      // below the sequence length could never accumulate, so fall back to
      // the full budget (fair sharing requires chunked prefill).
      per_group_cap = max_batch_tokens;
    }
    per_group_cap =
        std::min(std::max<int64_t>(1, per_group_cap),
                 static_cast<int64_t>(budget.remaining_token_budget));
    budget.dp_group_token_caps.assign(static_cast<size_t>(dp_size),
                                      static_cast<size_t>(per_group_cap));
    budget.dp_group_token_used.assign(static_cast<size_t>(dp_size), 0);
  }

  // Strategy-driven scheduling
  policy_->schedule(state, budget, finished);

  // Finalize
  if (!finished.empty()) {
    response_processor_->process_completed_requests(finished);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(running_requests_,
                           running_sequences_,
                           running_sequences_budgets_,
                           kv_cache_manager_->get_swap_block_transfer_infos());

  bool is_batches_empty = std::all_of(
      batches.begin(), batches.end(), [](const Batch& b) { return b.empty(); });
  if (!is_batches_empty) {
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
    kv_cache_manager_->transfer_blocks(batches);
  } else {
    kv_cache_manager_->transfer_blocks();
  }

  policy_->report_metrics(
      state, timer.elapsed_seconds(), budget.num_preempted_requests);
  return batches;
}

SchedulerState ContinuousScheduler::make_state() {
  return SchedulerState{
      .prefill_queue = *prefill_queue_,
      .chunk_queue = *chunk_queue_,
      .decode_queue = *decode_queue_,
      .unified_queue = unified_queue_,
      .decode_restore_waiting = decode_restore_waiting_,
      .running_requests = running_requests_,
      .running_sequences = running_sequences_,
      .running_sequences_budgets = running_sequences_budgets_,
      .kv_cache_manager = kv_cache_manager_,
      .profile_manager = profile_manager_.get(),
      .response_processor = response_processor_.get(),
      .model_args = engine_->model_args(),
      .last_step_prefill = last_step_prefill_,
      .options = options_,
      .min_speculative_tokens_required = min_speculative_tokens_required_,
      .enable_prefix_cache = enable_prefix_cache_,
      .has_linear_attention_layers = has_linear_attention_layers_,
  };
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
  if (!options_.enable_schedule_overlap()) {
    // get a new batch of requests
    std::vector<Batch> batch = schedule_request(timeout);
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (all_empty) {
      return;
    }

    engine_->step(batch);

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

void ContinuousScheduler::process_batch_output(bool enable_schedule_overlap) {
  std::vector<Sequence*>& to_be_processed_sequences =
      enable_schedule_overlap ? last_running_sequences_ : running_sequences_;
  std::vector<std::shared_ptr<Request>>& to_be_processed_requests =
      enable_schedule_overlap ? last_running_requests_ : running_requests_;
  // Beam search may replace Sequence objects inside SequencesGroup.
  // Always refresh the sequence pointers from requests before dereferencing.
  refresh_sequences_from_requests(to_be_processed_requests,
                                  to_be_processed_sequences);
  scheduler_metrics_->update(to_be_processed_sequences);

  std::vector<std::shared_ptr<Request>> stream_requests;
  stream_requests.reserve(to_be_processed_requests.size());
  // process request output in batch
  for (const auto& request : to_be_processed_requests) {
    // ignore cancelled/finished requests when enable_schedule_overlap.
    if (options_.enable_schedule_overlap()) {
      if (request->state().stream) {
        if (request->cancelled()) {
          continue;
        }
        if (request->error_status().has_value()) {
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
    } else if (request->state().stream &&
               !request->error_status().has_value()) {
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

}  // namespace xllm

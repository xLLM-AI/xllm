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

BatchMode resolve_batch_mode(const ContinuousScheduler::Options& options) {
  BatchMode mode;
  mode.priority_strategy = options.priority_strategy();
  mode.enable_chunked_prefill = options.enable_chunked_prefill();
  mode.enable_mix_batch =
      ::xllm::SchedulerConfig::get_instance().enable_mix_batch();

  // multi_slo_and_prio requires chunked prefill.
  if (mode.priority_strategy == "multi_slo_and_prio") {
    mode.enable_chunked_prefill = true;
  }

  // CP/MTP: prefill cannot mix with decode in the same batch.
  if (options.cp_size() > 1 || options.num_speculative_tokens() > 0) {
    mode.enable_mix_batch = false;
  }

  // No chunked prefill: prefill occupies the full batch exclusively.
  if (!mode.enable_chunked_prefill) {
    mode.enable_mix_batch = false;
  }

  return mode;
}

ContinuousScheduler::ContinuousScheduler(Engine* engine, const Options& options)
    : options_(options),
      batch_mode_(resolve_batch_mode(options)),
      engine_(engine),
      request_queue_(::xllm::RecConfig::get_instance().request_queue_size()) {
  CHECK(engine_ != nullptr);

  kv_cache_manager_ = engine_->block_manager_pool();
  CHECK(kv_cache_manager_ != nullptr);

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

  kv_cache_manager_->prefetch_from_storage(request);

  if (request_queue_.write(request)) {
    return true;
  }

  return false;
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

std::vector<Batch> ContinuousScheduler::prepare_batch() {
  Timer timer;
  auto state = make_state();

  // Common phases (strategy-independent)
  policy_->drain_request_queue(state, request_queue_);
  auto finished = policy_->collect_finished(state);

  // Initialize budget
  ScheduleBudget budget;
  budget.estimate_latency = profile_manager_->get_constant_overhead();
  budget.remaining_token_budget = options_.enable_profile_token_budget()
                                      ? profile_manager_->get_token_budget()
                                      : options_.max_tokens_per_batch();
  budget.remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  budget.latency_budget = options_.max_global_tpot_ms();
  budget.num_preempted_requests = 0;

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
      .running_requests = running_requests_,
      .running_sequences = running_sequences_,
      .running_sequences_budgets = running_sequences_budgets_,
      .kv_cache_manager = kv_cache_manager_,
      .profile_manager = profile_manager_.get(),
      .response_processor = response_processor_.get(),
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
  const size_t total_to_preempt =
      running_requests_.size() + decode_queue_->size() + chunk_queue_->size();
  if (total_to_preempt == 0) {
    return;
  }

  LOG(INFO) << "Preempting " << total_to_preempt
            << " running requests for pause";

  size_t preempted_count = 0;

  // Preempt a single request: free its KV cache and move it back to the
  // prefill queue so it will be re-prefilled on resume.
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

    // Push back to prefill queue (will need re-prefill on resume)
    prefill_queue_->push(request);

    preempted_count++;
  };

  // 1. Requests selected into the current batch.
  for (auto& request : running_requests_) {
    preempt_one(request);
  }

  // 2. Chunked prefill continuations in the chunk queue.
  while (!chunk_queue_->empty()) {
    preempt_one(chunk_queue_->top());
    chunk_queue_->pop_top();
  }

  // 3. Active decoding requests still waiting in the decode queue. These were
  // pushed back to decode_queue_ at the start of the step but were not selected
  // into running_requests_ (e.g. budget exhausted, or this step scheduled
  // prefill so decode was skipped). They still hold KV cache and must be
  // preempted as well.
  while (!decode_queue_->empty()) {
    preempt_one(decode_queue_->top());
    decode_queue_->pop_top();
  }

  // Clear running state
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  LOG(INFO) << "Preempted " << preempted_count
            << " requests, KV cache freed, moved to prefill queue";
}

void ContinuousScheduler::abort_all_running_requests() {
  const size_t total_to_abort =
      running_requests_.size() + decode_queue_->size() + chunk_queue_->size();
  if (total_to_abort == 0) {
    return;
  }

  LOG(INFO) << "Aborting " << total_to_abort << " running requests";

  size_t aborted_count = 0;

  // Abort a single request: free its KV cache and notify the client. Unlike
  // KEEP, aborted requests are NOT pushed back to the prefill queue.
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

  // 2. Chunked prefill continuations in the chunk queue.
  while (!chunk_queue_->empty()) {
    abort_one(chunk_queue_->top());
    chunk_queue_->pop_top();
  }

  // 3. Active decoding requests still waiting in the decode queue.
  while (!decode_queue_->empty()) {
    abort_one(decode_queue_->top());
    decode_queue_->pop_top();
  }

  // Clear running state.
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  LOG(INFO) << "Aborted " << aborted_count
            << " requests, KV cache freed (not rescheduled)";
}

}  // namespace xllm

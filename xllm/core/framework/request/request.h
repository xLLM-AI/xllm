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

#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "request_base.h"
#include "request_state.h"
#include "sequences_group.h"
#include "stopping_checker.h"
#include "util/threadpool.h"

namespace xllm {

enum class Urgency { STARVED = 2, URGENT = 1, NORMAL = 0 };

class Request : public RequestBase {
 public:
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const RequestState& state,
          const std::string& service_request_id = "",
          const std::string& source_xservice_addr = "");

  bool finished() const;

  std::vector<std::unique_ptr<Sequence>>& sequences() {
    return sequences_group_->sequences();
  }
  bool expand_sequences(bool share_prefix = true);

  SequencesGroup* sequence_group() { return sequences_group_.get(); }

  void set_cancel();

  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  RequestOutput generate_output(const Tokenizer& tokenizer,
                                ThreadPool* thread_pool = nullptr);

  void handle_last_token();

  bool last_token_handled() const { return state_.handle_last_token_done; }

  size_t total_num_blocks();

  void set_preempted() { state_.preempted = true; }

  bool preempted() const { return state_.preempted; }

  void log_statistic(double total_latency);

  void log_error_statistic(Status status);

  absl::Time created_time() const { return created_time_; }

  int32_t get_deadline_ms() const { return deadline_ms_; }

  void set_deadline_ms() {
    auto& sequence = sequences()[0];
    // w/o first token buffer
    if (sequence->is_prefill_stage()) {
      deadline_ms_ = ttft_slo_ms();
    } else {
      // deadline_ms_ =
      //     std::min(static_cast<int32_t>(
      //                  sequence->time_to_first_token_latency_seconds() *
      //                  1000),
      //              ttft_slo_ms()) +
      //     (sequence->num_tokens() - sequence->num_prompt_tokens()) *
      //         tpot_slo_ms();

      // only optimize for slo attainment
      deadline_ms_ = sequence->time_to_first_token_latency_seconds() * 1000 +
                     (sequence->num_tokens() - sequence->num_prompt_tokens()) *
                         tpot_slo_ms();
    }
    // w/ first token buffer
    // deadline_ms_ = ttft_slo_ms_ + (sequence->num_tokens() -
    // sequence->num_prompt_tokens()) * tpot_slo_ms_;
  }

  int32_t get_remaining_time() const {
    return get_deadline_ms() - get_elapsed_time_ms();
  }

  void set_elapsed_time_ms() {
    elapsed_time_ms_ = static_cast<int32_t>(
        absl::ToDoubleSeconds(absl::Now() - created_time_) * 1000);
  }
  int32_t get_elapsed_time_ms() const { return elapsed_time_ms_; }

  const bool offline() const { return state_.scheduler_param.offline; }
  const RequestPriority priority() const {
    return state_.scheduler_param.priority;
  }
  // time to last token (end-to-end latency)
  const int32_t ttlt_slo_ms() const {
    return state_.scheduler_param.ttlt_slo_ms;
  }
  const int32_t ttft_slo_ms() const {
    return state_.scheduler_param.ttft_slo_ms;
  }
  const int32_t tpot_slo_ms() const {
    return state_.scheduler_param.tpot_slo_ms;
  }
  const int32_t tpot_priority_weight() const {
    return state_.scheduler_param.tpot_priority_weight *
           state_.scheduler_param.priority_weight;
  }
  const int32_t ttft_priority_weight() const {
    return state_.scheduler_param.ttft_priority_weight *
           state_.scheduler_param.priority_weight;
  }
  const int32_t ttlt_priority_weight() const {
    return state_.scheduler_param.ttlt_priority_weight *
           state_.scheduler_param.priority_weight;
  }

  void set_urgency(Urgency urgency) { urgency_ = urgency; }
  Urgency urgency() const { return urgency_; }

  void set_starved(bool starved) { starved_ = starved; }
  bool is_starved() const { return starved_; }

  RequestState& state() { return state_; }
  size_t best_of() const { return state_.best_of; }
  void update_connection_status();

  bool check_beam_search() const {
    return state_.sampling_param.beam_width > 1;
  }

  bool is_chunked_prefill_stage() const {
    return sequences_group_->is_chunked_prefill_stage();
  }

  void record_num_prefix_cache_tokens();

  void record_num_prefix_cache_tokens(size_t num_prefix_cache_tokens);

  size_t num_prefix_cache_tokens() const { return num_prefix_cache_tokens_; }

 private:
  RequestState state_;
  // list of sequences to generate completions for the prompt
  // use deque instead of vector to avoid no-copy move for Sequence
  //  std::deque<Sequence> sequences;
  std::unique_ptr<SequencesGroup> sequences_group_;

  std::atomic<bool> cancelled_{false};

  int32_t elapsed_time_ms_ = 0;

  int32_t deadline_ms_ = 0;

  Urgency urgency_ = Urgency::NORMAL;

  bool starved_ = false;

  size_t num_prefix_cache_tokens_ = 0;

  void create_sequences_group();
};

}  // namespace xllm

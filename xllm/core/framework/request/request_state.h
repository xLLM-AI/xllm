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

#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/sampling/sampling_params.h"
#include "rec_type.h"
#include "request_output.h"
#include "sample_slot.h"
#include "stopping_checker.h"

namespace xllm {

using OutputFunc = std::function<bool(const RequestOutput& output)>;
using OutputsFunc =
    std::function<std::vector<bool>(const std::vector<RequestOutput>& outputs)>;

class Call;

enum class RequestPriority { DEFAULT = 0, HIGH = 1, NORMAL = 2, LOW = 3 };

struct SchedulerParam {
  bool offline = false;
  int32_t ttlt_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t ttft_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t tpot_slo_ms = std::numeric_limits<int32_t>::max();
  int32_t tpot_priority_weight = 1;
  int32_t ttft_priority_weight = 1;
  int32_t ttlt_priority_weight = 1;
  int32_t priority_weight = 1;
  RequestPriority priority = RequestPriority::NORMAL;
};

class RequestState final {
 public:
  RequestState() {}

  RequestState(std::string prompt,
               std::vector<int32_t> prompt_tokens,
               RequestSamplingParam sampling_param,
               SchedulerParam scheduler_param,
               StoppingChecker stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "",
               std::optional<Call*> call = std::nullopt);

  RequestState(std::string prompt,
               std::vector<int32_t> prompt_tokens,
               torch::Tensor input_embedding,
               RequestSamplingParam sampling_param,
               StoppingChecker stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "");

  RequestState(std::string prompt,
               std::vector<int32_t> prompt_tokens,
               MMData mm_data,
               RequestSamplingParam sampling_param,
               StoppingChecker stopping_checker,
               size_t seq_capacity,
               size_t n,
               size_t best_of,
               bool logprobs,
               bool stream,
               bool echo,
               bool skip_special_tokens,
               bool enable_schedule_overlap,
               const OutputFunc& output_func,
               const OutputsFunc& outputs_func,
               const std::string& decode_address = "");

  // for profiling run, only provide prompt tokens
  RequestState(const std::vector<int32_t>& prompt_tokens);

  // RequestState owns the heavy request payload (prompt, prompt_tokens,
  // mm_data, sample_slots, callbacks). An implicit copy is almost always an
  // accidental deep copy, so the type is move-only. Use clone() for the rare
  // intentional copy (e.g. a benchmark/test that reuses a template state). This
  // also makes the std::move(const&) foot-gun a compile error rather than a
  // silent copy.
  RequestState(RequestState&&) = default;
  RequestState& operator=(RequestState&&) = default;

  // Explicit deep copy. Prefer moving; only clone when a genuine second owner
  // is required.
  RequestState clone() const { return RequestState(*this); }

 private:
  // Non-public so external code cannot copy implicitly; clone() uses it.
  RequestState(const RequestState&) = default;
  RequestState& operator=(const RequestState&) = delete;

 public:
  // sampling parameters
  RequestSamplingParam sampling_param;

  // scheduling parameters
  SchedulerParam scheduler_param;

  // stopping criteria
  StoppingChecker stopping_checker;

  std::string prompt;

  std::vector<int32_t> prompt_tokens;

  bool stream = false;

  // max tokens for a seq
  size_t seq_capacity;

  size_t n;

  size_t best_of;

  bool echo = false;

  bool skip_special_tokens = true;

  bool include_stop_str_in_output = false;

  OutputFunc output_func;

  // function to call when batch outputs is generated in disagg pd mode,
  // decode will send the batch outputs to prefill.
  OutputsFunc outputs_func;

  // decode address.
  std::string decode_address;

  torch::Tensor input_embedding;

  // multimodal
  MMData mm_data;

  // whether to return log probabilities for output token.
  bool logprobs;

  bool enable_schedule_overlap = false;

  bool is_graph_warmup = false;

  RecType rec_type = RecType::kNone;

  int32_t bos_token_id = 0;

  // The thread id of the thread pool in the response handler to ensure that
  // stream responses for the same request are executed sequentially during
  // multi-threaded stream processing.
  int32_t response_thread_id = -1;

  bool preempted = false;

  // This will be used in enable_scheduler_overlap
  bool handle_last_token_done = false;

  std::optional<Call*> call_;

  std::vector<SampleSlot> sample_slots;

  std::shared_ptr<const JsonObjectGrammar> json_object_grammar;
  bool json_reasoning_enabled = false;
};

}  // namespace xllm

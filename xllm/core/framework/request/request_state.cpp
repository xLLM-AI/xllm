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

#include "request_state.h"

#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "api_service/call.h"

namespace xllm {

RequestState::RequestState(const std::string& prompt,
                           const std::vector<int32_t>& prompt_tokens,
                           const RequestSamplingParam& sampling_param,
                           const SchedulerParam& scheduler_param,
                           const StoppingChecker& stopping_checker,
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
                           const std::string& decode_address,
                           std::optional<Call*> call)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      sampling_param(std::move(sampling_param)),
      scheduler_param(std::move(scheduler_param)),
      stopping_checker(std::move(stopping_checker)),
      seq_capacity(seq_capacity),
      n(n),
      best_of(best_of),
      logprobs(logprobs),
      stream(stream),
      echo(echo),
      skip_special_tokens(skip_special_tokens),
      enable_schedule_overlap(enable_schedule_overlap),
      output_func(output_func),
      outputs_func(outputs_func),
      decode_address(decode_address),
      call_(call) {
  if (best_of < n) {
    LOG(FATAL) << "best_of must greater than n.";
  }
}

RequestState::RequestState(const std::string& prompt,
                           const std::vector<int32_t>& prompt_tokens,
                           torch::Tensor input_embedding,
                           const RequestSamplingParam& sampling_param,
                           const StoppingChecker& stopping_checker,
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
                           const std::string& decode_address)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      input_embedding(input_embedding),
      sampling_param(std::move(sampling_param)),
      stopping_checker(std::move(stopping_checker)),
      seq_capacity(seq_capacity),
      n(n),
      best_of(best_of),
      logprobs(logprobs),
      stream(stream),
      echo(echo),
      skip_special_tokens(skip_special_tokens),
      enable_schedule_overlap(enable_schedule_overlap),
      output_func(output_func),
      outputs_func(outputs_func),
      decode_address(decode_address) {
  if (best_of < n) {
    LOG(FATAL) << "best_of must greater than n.";
  }
}

RequestState::RequestState(const std::string& prompt,
                           const std::vector<int32_t>& prompt_tokens,
                           const MMData& mm_data,
                           const RequestSamplingParam& sampling_param,
                           const StoppingChecker& stopping_checker,
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
                           const std::string& decode_address)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      mm_data(std::move(mm_data)),
      sampling_param(std::move(sampling_param)),
      stopping_checker(std::move(stopping_checker)),
      seq_capacity(seq_capacity),
      n(n),
      best_of(best_of),
      logprobs(logprobs),
      stream(stream),
      echo(echo),
      skip_special_tokens(skip_special_tokens),
      enable_schedule_overlap(enable_schedule_overlap),
      output_func(output_func),
      outputs_func(outputs_func),
      decode_address(decode_address) {
  if (best_of < n) {
    LOG(FATAL) << "best_of must greater than n.";
  }
}

RequestState::RequestState(const std::vector<int32_t>& prompt_tokens)
    : prompt_tokens(std::move(prompt_tokens)),
      seq_capacity(prompt_tokens.size() + 1) {}

}  // namespace xllm

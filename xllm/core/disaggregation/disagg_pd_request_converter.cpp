/* Copyright 2026 The xLLM Authors.

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

#include "core/disaggregation/disagg_pd_request_converter.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xllm {

RequestSamplingParam make_disagg_sampling_param(
    const proto::DisaggRequest& req) {
  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = req.frequency_penalty();
  sampling_param.presence_penalty = req.presence_penalty();
  sampling_param.repetition_penalty = req.repetition_penalty();
  sampling_param.temperature = req.temperature();
  sampling_param.top_p = req.top_p();
  sampling_param.min_p = req.min_p();
  if (req.has_seed()) {
    sampling_param.seed = req.seed();
  }
  sampling_param.min_tokens = req.min_tokens();
  sampling_param.vocab_size = req.vocab_size();
  if (!req.logit_bias().empty()) {
    sampling_param.logit_bias.reserve(req.logit_bias_size());
  }
  for (const auto& [token, bias] : req.logit_bias()) {
    sampling_param.logit_bias.emplace(token, bias);
  }
  if (!req.allowed_token_ids().empty()) {
    sampling_param.allowed_token_ids = std::vector<int32_t>(
        req.allowed_token_ids().begin(), req.allowed_token_ids().end());
  }
  sampling_param.all_stop_token_ids.assign(req.all_stop_token_ids().begin(),
                                           req.all_stop_token_ids().end());
  sampling_param.bad_words_token_ids.reserve(req.bad_words_token_ids_size());
  for (const auto& word : req.bad_words_token_ids()) {
    sampling_param.bad_words_token_ids.emplace_back(word.seq_tokens().begin(),
                                                    word.seq_tokens().end());
  }
  sampling_param.top_k = req.top_k();
  sampling_param.logprobs = req.logprobs();
  sampling_param.top_logprobs = req.top_logprobs();
  sampling_param.is_embeddings = req.is_embeddings();
  sampling_param.json_object = req.json_object();
  return sampling_param;
}

SchedulerParam make_disagg_scheduler_param(const proto::DisaggRequest& req) {
  SchedulerParam scheduler_param;
  scheduler_param.offline = req.offline();
  scheduler_param.priority = static_cast<RequestPriority>(req.priority());
  if (!scheduler_param.offline) {
    scheduler_param.ttft_slo_ms = req.ttft_slo_ms();
    scheduler_param.tpot_slo_ms = req.tpot_slo_ms();
    scheduler_param.ttlt_slo_ms = req.ttlt_slo_ms();
    scheduler_param.tpot_priority_weight = req.tpot_priority_weight();
    scheduler_param.ttft_priority_weight = req.ttft_priority_weight();
    scheduler_param.ttlt_priority_weight = req.ttlt_priority_weight();
    scheduler_param.priority_weight = req.priority_weight();
  }
  return scheduler_param;
}

StoppingChecker make_disagg_stopping_checker(const proto::DisaggRequest& req) {
  std::unordered_set<int32_t> stop_tokens;
  if (!req.stop_token_ids().empty()) {
    stop_tokens.reserve(req.stop_token_ids_size());
  }
  stop_tokens.insert(req.stop_token_ids().begin(), req.stop_token_ids().end());
  std::vector<std::vector<int32_t>> stop_sequences;
  std::vector<std::string> stop_strings;
  stop_sequences.reserve(req.stop_sequences_size());
  stop_strings.reserve(req.stop_sequences_size());
  for (const auto& stop_sequence : req.stop_sequences()) {
    stop_sequences.emplace_back(stop_sequence.seq_tokens().begin(),
                                stop_sequence.seq_tokens().end());
    stop_strings.emplace_back(stop_sequence.stop_string());
  }
  return StoppingChecker(req.max_tokens(),
                         req.max_context_len(),
                         req.eos_token_id(),
                         req.ignore_eos(),
                         std::move(stop_tokens),
                         std::move(stop_sequences),
                         std::move(stop_strings),
                         req.min_tokens());
}

}  // namespace xllm

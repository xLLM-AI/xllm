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

#include "core/disaggregation/pd_request_converter.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/request/request.h"

namespace xllm {

// Keep request-to-wire conversion separate from dispatch admission and retries.
void fill_pd_request(Request& request, proto::DisaggRequest* req) {
  RequestState& state = request.state();
  req->set_req_id(request.request_id());
  req->set_reservation_id(state.pd_reservation_id);
  req->set_service_req_id(request.service_request_id());
  req->set_source_xservice_addr(request.source_xservice_addr());
  req->set_tokens_num(state.prompt_tokens.size());
  req->set_prompt(state.prompt);
  req->set_response_type(state.response_type);
  req->set_model(state.response_model);
  req->mutable_tools()->Assign(state.response_tools.begin(),
                               state.response_tools.end());
  req->set_tool_choice(state.response_tool_choice);
  *req->mutable_chat_template_kwargs() = state.response_chat_template_kwargs;
  *req->mutable_stream_options() = state.response_stream_options;
  ADD_VECTOR_TO_PROTO(req->mutable_prompt_tokens(), state.prompt_tokens);
  req->set_stream(state.stream);
  req->set_x_request_id(request.x_request_id());
  req->set_x_request_time(request.x_request_time());
  req->set_seq_capacity(state.seq_capacity);
  req->set_max_tokens(state.stopping_checker.get_max_generated_tokens());
  req->set_max_context_len(state.stopping_checker.get_max_context_len());
  req->set_ignore_eos(state.stopping_checker.get_ignore_eos());
  req->set_eos_token_id(state.stopping_checker.get_eos_token());
  if (!state.stopping_checker.get_stop_tokens().empty()) {
    ADD_VECTOR_TO_PROTO(req->mutable_stop_token_ids(),
                        state.stopping_checker.get_stop_tokens());
  }
  const auto& stop_sequences = state.stopping_checker.get_stop_sequences();
  const auto& stop_strings = state.stopping_checker.get_stop_strings();
  for (size_t index = 0; index < stop_sequences.size(); ++index) {
    auto* proto_seq = req->add_stop_sequences();
    ADD_VECTOR_TO_PROTO(proto_seq->mutable_seq_tokens(), stop_sequences[index]);
    if (index < stop_strings.size()) {
      proto_seq->set_stop_string(stop_strings[index]);
    }
  }
  req->set_n(state.n);
  req->set_best_of(state.best_of);
  req->set_frequency_penalty(state.sampling_param.frequency_penalty);
  req->set_presence_penalty(state.sampling_param.presence_penalty);
  req->set_repetition_penalty(state.sampling_param.repetition_penalty);
  req->set_temperature(state.sampling_param.temperature);
  req->set_top_p(state.sampling_param.top_p);
  req->set_min_p(state.sampling_param.min_p);
  const RequestSamplingParam& sampling = state.sampling_param;
  if (sampling.seed.has_value()) {
    req->set_seed(*sampling.seed);
  }
  req->set_min_tokens(sampling.min_tokens);
  req->set_vocab_size(sampling.vocab_size);
  req->mutable_logit_bias()->insert(sampling.logit_bias.begin(),
                                    sampling.logit_bias.end());
  if (sampling.allowed_token_ids.has_value()) {
    ADD_VECTOR_TO_PROTO(req->mutable_allowed_token_ids(),
                        sampling.allowed_token_ids.value());
  }
  ADD_VECTOR_TO_PROTO(req->mutable_all_stop_token_ids(),
                      sampling.all_stop_token_ids);
  for (const auto& word : sampling.bad_words_token_ids) {
    ADD_VECTOR_TO_PROTO(req->add_bad_words_token_ids()->mutable_seq_tokens(),
                        word);
  }
  req->set_top_k(state.sampling_param.top_k);
  req->set_logprobs(state.sampling_param.logprobs);
  req->set_top_logprobs(state.sampling_param.top_logprobs);
  req->set_is_embeddings(state.sampling_param.is_embeddings);
  req->set_echo(state.echo);
  req->set_skip_special_tokens(state.skip_special_tokens);
  req->set_include_stop_str_in_output(state.include_stop_str_in_output);
  req->set_json_object(state.sampling_param.json_object);
  req->set_json_reasoning_enabled(state.json_reasoning_enabled);
}

RequestSamplingParam make_pd_sampling_param(const proto::DisaggRequest& req) {
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

SchedulerParam make_pd_scheduler_param(const proto::DisaggRequest& req) {
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

StoppingChecker make_pd_stopping_checker(const proto::DisaggRequest& req) {
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

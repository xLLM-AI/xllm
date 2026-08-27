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

#include "vlm_request_factory.h"

#include <glog/logging.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "framework/tokenizer/tokenizer.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {

VLMRequestFactory::VLMRequestFactory(MultimodalProcessorBase* processor,
                                     JinjaChatTemplate* chat_template,
                                     const Tokenizer* tokenizer,
                                     const ModelArgs* model_args,
                                     const Options* options,
                                     RateLimiter* rate_limiter)
    : processor_(processor),
      chat_template_(chat_template),
      tokenizer_(tokenizer),
      model_args_(model_args),
      options_(options),
      rate_limiter_(rate_limiter) {
  CHECK(processor_ != nullptr);
  CHECK(chat_template_ != nullptr);
  CHECK(tokenizer_ != nullptr);
  CHECK(model_args_ != nullptr);
  CHECK(options_ != nullptr);
  CHECK(rate_limiter_ != nullptr);
}

std::shared_ptr<Request> VLMRequestFactory::create(std::string prompt,
                                                   MMData mm_data,
                                                   const RequestParams& sp,
                                                   OutputCallback callback) {
  // Guard the rate-limit slot acquired at the service entry. build_request
  // installs its own guard, so we dismiss ours before forwarding.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  if (prompt.empty() && mm_data.empty()) {
    LOG(ERROR) << "Prompt and multimodal data cannot be both empty.";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt and multimodal data are both empty.");
    return nullptr;
  }

  std::vector<int32_t> prompt_tokens;
  if (!processor_->process_prompt(prompt, mm_data, prompt_tokens)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to process prompt.");
    return nullptr;
  }

  rate_limit_guard.dismiss();
  return build_request(std::move(prompt),
                       std::move(prompt_tokens),
                       std::move(mm_data),
                       sp,
                       std::move(callback));
}

std::shared_ptr<Request> VLMRequestFactory::build_request(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    OutputCallback callback) {
  // Guard the rate-limit slot acquired at the service entry. Any early
  // return below releases it; success path dismisses right before Request
  // takes ownership.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  const int32_t max_context_len = model_args_->max_position_embeddings();
  int32_t prompt_token_limit = max_context_len;
  if (!options_->enable_chunked_prefill()) {
    prompt_token_limit =
        std::min(prompt_token_limit, options_->max_tokens_per_batch());
  }
  if (prompt_tokens.size() >= static_cast<size_t>(prompt_token_limit)) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens, TODO: add image token size as well.
  size_t capacity = prompt_tokens.size() + max_tokens +
                    options_->num_speculative_tokens() + /*bonus_token*/ 1;
  if (options_->enable_schedule_overlap()) {
    capacity += options_->num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param = sp.to_sampling_param(best_of);
  // Model-aware complement to verify_params' 2000 cap: an oversized top-k
  // would throw inside the sampler and take the whole batch down with it.
  if (const auto error =
          sampling_param.top_logprobs_vocab_error(model_args_->vocab_size())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, *error);
    return nullptr;
  }

  std::optional<StoppingChecker> stopping_checker =
      build_stopping_checker(sp, max_tokens, max_context_len, callback);
  if (!stopping_checker.has_value()) {
    return nullptr;
  }

  // results cannot be streamed when best_of != n
  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
                         std::move(stopping_checker.value()),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_->enable_schedule_overlap(),
                         callback,
                         nullptr);
  req_state.include_stop_str_in_output = sp.include_stop_str_in_output;

  rate_limit_guard.dismiss();
  // add one sequence, rest will be added by scheduler
  return std::make_shared<Request>(sp.request_id,
                                   sp.x_request_id,
                                   sp.x_request_time,
                                   std::move(req_state),
                                   sp.service_request_id,
                                   sp.source_xservice_addr,
                                   rate_limiter_);
}

std::optional<StoppingChecker> VLMRequestFactory::build_stopping_checker(
    const RequestParams& sp,
    uint32_t max_tokens,
    int32_t max_context_len,
    const OutputCallback& callback) {
  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else if (!sp.ignore_eos) {
    stop_tokens = model_args_->stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int32_t> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return std::nullopt;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  return StoppingChecker(max_tokens,
                         max_context_len,
                         model_args_->eos_token_id(),
                         sp.ignore_eos,
                         std::move(stop_tokens),
                         std::move(stop_sequences));
}

std::shared_ptr<Request> VLMRequestFactory::create(
    std::vector<Message> messages,
    const RequestParams& sp,
    std::string payload,
    OutputCallback callback) {
  // Guard the rate-limit slot acquired at the service entry. The next hop
  // (create(prompt, ...)) installs its own guard, so we dismiss ours before
  // forwarding.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  static MMInputTransfer mm_input_transfer;

  MMInput mm_inputs(std::move(payload));
  MMErrCode code = mm_input_transfer.trans(messages, mm_inputs);
  if (code != MMErrCode::SUCCESS) {
    std::string error_message = MMErrToString(code);
    LOG(ERROR) << error_message;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, error_message);
    return nullptr;
  }

  MMData mm_data;
  if (!mm_inputs.empty() &&
      !processor_->process_multimodal(mm_inputs, mm_data)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to process multimodal input.");
    return nullptr;
  }

  Timer timer;
  std::optional<std::string> prompt =
      chat_template_->apply(messages, sp.tools, sp.chat_template_kwargs);
  if (!prompt.has_value()) {
    std::string error_message = "Failed to construct prompt from messages";
    LOG(ERROR) << error_message;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, error_message);
    return nullptr;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  rate_limit_guard.dismiss();
  return create(
      std::move(prompt.value()), std::move(mm_data), sp, std::move(callback));
}

}  // namespace xllm

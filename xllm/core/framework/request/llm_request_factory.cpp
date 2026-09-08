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

#include "llm_request_factory.h"

#include <glog/logging.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/macros.h"
#include "common/metrics.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/service_config.h"
#include "framework/request/request_state.h"
#include "framework/request/stopping_checker.h"
#include "framework/tokenizer/tokenizer.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

bool get_enable_thinking(const nlohmann::json& chat_template_kwargs) {
  const bool default_value =
      !ModelConfig::get_instance().reasoning_parser().empty();
  if (!chat_template_kwargs.contains("enable_thinking") &&
      !chat_template_kwargs.contains("thinking")) {
    return default_value;
  }
  bool enabled = false;
  for (const char* key : {"enable_thinking", "thinking"}) {
    const auto it = chat_template_kwargs.find(key);
    if (it != chat_template_kwargs.end() && it->is_boolean()) {
      enabled = enabled || it->get<bool>();
    }
  }
  return enabled;
}

}  // namespace

LLMRequestFactory::LLMRequestFactory(const Tokenizer* tokenizer,
                                     const ChatTemplate* chat_template,
                                     const ModelArgs* model_args,
                                     const Options* options,
                                     RateLimiter* rate_limiter,
                                     std::string task_type,
                                     RpcResponseHandler rpc_response_handler)
    : tokenizer_(tokenizer),
      chat_template_(chat_template),
      model_args_(model_args),
      options_(options),
      rate_limiter_(rate_limiter),
      task_type_(std::move(task_type)),
      rpc_response_handler_(std::move(rpc_response_handler)) {
  CHECK(tokenizer_ != nullptr);
  CHECK(chat_template_ != nullptr);
  CHECK(model_args_ != nullptr);
  CHECK(options_ != nullptr);
  CHECK(rate_limiter_ != nullptr);
}

std::shared_ptr<const JsonObjectGrammar>
LLMRequestFactory::get_json_object_grammar(bool reasoning_enabled,
                                           std::string* error) {
  std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
  std::shared_ptr<const JsonObjectGrammar>& grammar =
      reasoning_enabled ? json_reasoning_grammar_ : json_object_grammar_;
  if (grammar == nullptr) {
    grammar =
        JsonObjectGrammar::create_from_tokenizer(*tokenizer_,
                                                 model_args_->eos_token_id(),
                                                 model_args_->stop_token_ids(),
                                                 model_args_->vocab_size(),
                                                 reasoning_enabled,
                                                 error);
  }
  return grammar;
}

std::optional<std::vector<int>> LLMRequestFactory::encode_and_validate_prompt(
    const std::string& prompt,
    std::optional<std::vector<int>> prompt_tokens,
    int32_t max_context_len,
    const RequestParams& sp,
    const OutputCallback& callback) {
  // A request is valid as long as it carries either text or pre-tokenized
  // prompt tokens; pure-token input (no text) is a first-class input.
  const bool has_prompt_tokens = prompt_tokens.has_value();
  if (prompt.empty() && !has_prompt_tokens) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt is empty",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return std::nullopt;
  }

  // encode the prompt
  Timer timer;
  std::vector<int> local_prompt_tokens;
  if (has_prompt_tokens) {
    local_prompt_tokens = std::move(prompt_tokens.value());
  } else {
    if (!tokenizer_->encode(
            prompt, &local_prompt_tokens, sp.add_special_tokens)) {
      LOG(ERROR) << "Failed to encode prompt: " << prompt;
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to encode prompt",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return std::nullopt;
    }
  }
  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  // Validate directly-supplied prompt tokens against the vocabulary range to
  // avoid out-of-bounds embedding lookups. Encoded tokens are trusted, so only
  // scan when tokens were provided and the vocab range is known.
  const int64_t vocab_size = model_args_->vocab_size();
  if (has_prompt_tokens && vocab_size > 0) {
    const auto invalid_token =
        util::find_out_of_vocab_token(local_prompt_tokens, vocab_size);
    if (invalid_token.has_value()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Prompt token id out of vocabulary range: " +
                              std::to_string(invalid_token.value()),
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return std::nullopt;
    }
  }

  int32_t prompt_token_limit = max_context_len;
  if (!options_->enable_chunked_prefill()) {
    prompt_token_limit =
        std::min(prompt_token_limit, options_->max_tokens_per_batch());
  }
  if (local_prompt_tokens.size() >= prompt_token_limit) {
    LOG(ERROR) << "Prompt is too long: " << local_prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt is too long",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return std::nullopt;
  }

  return local_prompt_tokens;
}

RequestSamplingParam LLMRequestFactory::build_sampling_param(
    const RequestParams& sp,
    size_t best_of) const {
  // Shared field mapping (and the best_of > n logprobs rule) lives on
  // RequestParams; layer the LLM-specific rules on top.
  RequestSamplingParam sampling_param = sp.to_sampling_param(best_of);
  sampling_param.json_object =
      ServiceConfig::get_instance().enable_json_object_output() &&
      sp.response_format == ResponseFormatType::JSON_OBJECT;
  // Enforces the beam-search logprob requirements; see
  // RequestSamplingParam::enable_beam_search.
  sampling_param.enable_beam_search(sp.beam_width);
  // sampling_param.do_sample = sp.do_sample;
  return sampling_param;
}

std::optional<StoppingChecker> LLMRequestFactory::build_stopping_checker(
    const RequestParams& sp,
    uint32_t effective_max_tokens,
    int32_t max_context_len,
    const OutputCallback& callback) {
  std::unordered_set<int32_t> stop_tokens;
  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
  } else {
    stop_tokens = model_args_->stop_token_ids();
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> tmp_tokens;
      if (!tokenizer_->encode(s, &tmp_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence",
                            sp.service_request_id,
                            sp.source_xservice_addr);
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return std::nullopt;
      }
      stop_sequences.push_back(std::move(tmp_tokens));
    }
  }

  return StoppingChecker(effective_max_tokens,
                         max_context_len - options_->num_speculative_tokens(),
                         model_args_->eos_token_id(),
                         sp.ignore_eos,
                         std::move(stop_tokens),
                         std::move(stop_sequences));
}

bool LLMRequestFactory::validate_prompt_not_finished(
    const StoppingChecker& stopping_checker,
    const std::vector<int>& prompt_tokens,
    const RequestParams& sp,
    const OutputCallback& callback) const {
  if (task_type_ == "embed" || task_type_ == "mm_embed") {
    return true;
  }
  auto finish_reason =
      stopping_checker.check(prompt_tokens, prompt_tokens.size());
  if (finish_reason == FinishReason::NONE) {
    return true;
  }
  LOG(INFO) << " finish_reason " << finish_reason.to_string().value();
  CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                      "Invalid Prompt",
                      sp.service_request_id,
                      sp.source_xservice_addr);
  LOG(ERROR) << "Invalid Prompt EndWith Token_ID:"
             << prompt_tokens[prompt_tokens.size() - 1];
  return false;
}

bool LLMRequestFactory::apply_json_object_grammar(
    RequestState& req_state,
    const RequestParams& sp,
    std::optional<ChatTemplateGenerationMode> generation_mode,
    const OutputCallback& callback) {
  std::string grammar_error;
  bool reasoning_enabled = false;
  if (generation_mode.has_value()) {
    if (generation_mode == ChatTemplateGenerationMode::UNKNOWN) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "JSON object constraint requires a recognizable chat generation "
          "mode",
          sp.service_request_id,
          sp.source_xservice_addr);
      return false;
    }
    reasoning_enabled =
        generation_mode == ChatTemplateGenerationMode::REASONING;
  } else {
    reasoning_enabled = get_enable_thinking(sp.chat_template_kwargs);
  }
  req_state.json_object_grammar =
      get_json_object_grammar(reasoning_enabled, &grammar_error);
  if (req_state.json_object_grammar == nullptr) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Failed to initialize json_object constraint: " + grammar_error,
        sp.service_request_id,
        sp.source_xservice_addr);
    return false;
  }
  req_state.json_reasoning_enabled = reasoning_enabled;
  return true;
}

std::shared_ptr<Request> LLMRequestFactory::create(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback,
    std::optional<ChatTemplateGenerationMode> generation_mode) {
  // The caller (service_impl) has already incremented the rate limiter's
  // slot via is_limited() returning false. This guard releases it on any
  // early return below; we dismiss it right before Request takes ownership.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  const int32_t max_context_len = model_args_->max_position_embeddings();

  std::optional<std::vector<int>> encoded = encode_and_validate_prompt(
      prompt, std::move(prompt_tokens), max_context_len, sp, callback);
  if (!encoded.has_value()) {
    return nullptr;
  }
  std::vector<int> local_prompt_tokens = std::move(encoded.value());

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }
  uint32_t effective_max_tokens = max_tokens;
  if (sp.is_sample_request) {
    const uint32_t sample_slot_tokens =
        static_cast<uint32_t>(sp.sample_slots.size());
    if (sample_slot_tokens > effective_max_tokens) {
      effective_max_tokens = sample_slot_tokens;
    }
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  size_t capacity = local_prompt_tokens.size() + effective_max_tokens +
                    options_->num_speculative_tokens() + /*bouns_token*/ 1;
  if (options_->enable_schedule_overlap()) {
    capacity += options_->num_speculative_tokens() + 1;
  }

  const size_t best_of = sp.best_of.value_or(sp.n);
  RequestSamplingParam sampling_param = build_sampling_param(sp, best_of);
  // Model-aware complement to verify_params' 2000 cap: an oversized top-k
  // would throw inside the sampler and take the whole batch down with it.
  if (const auto error =
          sampling_param.top_logprobs_vocab_error(model_args_->vocab_size())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, *error);
    return nullptr;
  }
  const bool json_object = sampling_param.json_object;
  SchedulerParam scheduler_param = sp.to_scheduler_param();

  std::optional<StoppingChecker> stopping_checker = build_stopping_checker(
      sp, effective_max_tokens, max_context_len, callback);
  if (!stopping_checker.has_value()) {
    return nullptr;
  }
  if (!validate_prompt_not_finished(
          *stopping_checker, local_prompt_tokens, sp, callback)) {
    return nullptr;
  }

  bool stream = sp.streaming;
  // results cannot be streamed when best_of != n
  if (best_of != sp.n) {
    stream = false;
  }

  OutputsFunc batch_callback = nullptr;
  if (options_->enable_service_routing()) {
    // Capture a copy of the handler rather than `this`: the callback is stored
    // in RequestState and drained by the scheduler during shutdown, which
    // happens after this factory is destroyed. Copying keeps the callback
    // self-contained and avoids a use-after-free on the factory.
    batch_callback = [handler = rpc_response_handler_](
                         const std::vector<RequestOutput>& req_outputs) {
      for (const auto& req_output : req_outputs) {
        req_output.log_request_status();
      }
      return handler(req_outputs);
    };
  }

  RequestState req_state(std::move(prompt),
                         std::move(local_prompt_tokens),
                         std::move(sampling_param),
                         std::move(scheduler_param),
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
                         batch_callback,
                         sp.decode_address,
                         call);
  req_state.include_stop_str_in_output = sp.include_stop_str_in_output;
  if (json_object &&
      !apply_json_object_grammar(req_state, sp, generation_mode, callback)) {
    return nullptr;
  }
  req_state.sample_slots = sp.sample_slots;

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

std::shared_ptr<Request> LLMRequestFactory::create(
    const std::vector<Message>& messages,
    std::optional<std::vector<int>> prompt_tokens,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback) {
  // Guard the rate-limit slot the caller acquired via is_limited(). The
  // string-prompt overload installs its own guard once we forward there;
  // we dismiss ours right before that call so the slot is not released
  // twice.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  Timer timer;

  const std::optional<ChatTemplateRenderResult> render_result =
      chat_template_->apply_with_generation_mode(
          messages, sp.tools, sp.chat_template_kwargs);
  if (!render_result.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }

  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  rate_limit_guard.dismiss();
  return create(std::move(render_result->prompt),
                std::move(prompt_tokens),
                sp,
                call,
                callback,
                render_result->generation_mode);
}

}  // namespace xllm

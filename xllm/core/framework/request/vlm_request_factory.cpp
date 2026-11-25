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

#include "core/framework/request/vlm_request_factory.h"

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/model/vocab_validation.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "framework/tokenizer/tokenizer.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {

namespace {

bool get_enable_thinking(const nlohmann::json& kwargs) {
  if (!kwargs.contains("enable_thinking") && !kwargs.contains("thinking")) {
    return !ModelConfig::get_instance().reasoning_parser().empty();
  }
  bool enabled = false;
  for (const char* key : {"enable_thinking", "thinking"}) {
    const auto it = kwargs.find(key);
    if (it != kwargs.end() && it->is_boolean()) {
      enabled = enabled || it->get<bool>();
    }
  }
  return enabled;
}

// Extracts the final prompt from a single message, concatenating its text
// contents in order and skipping non-text (image/video/audio) contents.
//
// The offline "prompt as-is" contract expects the caller to hand in exactly
// one message whose text is the final prompt already assembled with the
// chat template (images may ride along as non-text contents). Multi-turn
// conversations must go through the chat template path instead; joining
// their texts would silently mash the roles together, so any other message
// shape returns std::nullopt and fails the request.
std::optional<std::string> join_message_texts(
    const std::vector<Message>& messages) {
  if (messages.size() != 1) {
    return std::nullopt;
  }
  const auto& message = messages.front();
  if (auto text = std::get_if<std::string>(&message.content)) {
    return *text;
  }
  if (auto contents = std::get_if<MMContentVec>(&message.content)) {
    std::string prompt;
    for (const auto& content : *contents) {
      if (content.type == "text") {
        prompt += content.text;
      }
    }
    return prompt;
  }
  return std::nullopt;
}

// Throws std::invalid_argument describing the violation when messages do not
// satisfy the offline "prompt as-is" contract (see join_message_texts).
// Offline-only: raised on the caller thread so pybind surfaces it to the
// Python caller at the call site, instead of returning an empty prompt that
// is hard to trace.
[[noreturn]] void throw_prompt_as_is_error(
    const std::vector<Message>& messages) {
  std::string roles;
  for (const auto& message : messages) {
    if (!roles.empty()) {
      roles += ", ";
    }
    roles += message.role;
  }
  throw std::invalid_argument(
      "use_prompt_as_is expects exactly 1 message holding the final prompt "
      "(images allowed as non-text contents), got " +
      std::to_string(messages.size()) + " messages [" + roles +
      "]; apply the chat template for multi-turn conversations");
}

}  // namespace

VLMRequestFactory::VLMRequestFactory(MultimodalProcessorBase* processor,
                                     ChatTemplate* chat_template,
                                     const Tokenizer* tokenizer,
                                     const ModelArgs* model_args,
                                     const Options* options,
                                     RateLimiter* rate_limiter,
                                     RpcResponseHandler rpc_response_handler)
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
  if (options_->enable_service_routing()) {
    CHECK(rpc_response_handler != nullptr);
    batch_callback_ = [handler = std::move(rpc_response_handler)](
                          const std::vector<RequestOutput>& outputs) {
      for (const auto& output : outputs) {
        output.log_request_status();
      }
      return handler(outputs);
    };
  }
}

std::shared_ptr<Request> VLMRequestFactory::create(
    std::string prompt,
    MMData mm_data,
    const RequestParams& sp,
    OutputCallback callback,
    std::chrono::steady_clock::time_point arrival_time) {
  return create(std::move(prompt),
                std::nullopt,
                std::move(mm_data),
                sp,
                std::nullopt,
                std::move(callback),
                std::nullopt,
                arrival_time);
}

std::shared_ptr<Request> VLMRequestFactory::create(
    std::string prompt,
    std::optional<std::vector<int32_t>> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback,
    std::optional<ChatTemplateGenerationMode> generation_mode,
    std::chrono::steady_clock::time_point arrival_time) {
  // Guard the rate-limit slot acquired at the service entry. build_request
  // installs its own guard, so we dismiss ours before forwarding.
  xllm::ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });

  if (prompt.empty() && mm_data.empty() && !prompt_tokens.has_value()) {
    LOG(ERROR) << "Prompt and multimodal data cannot be both empty.";
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt and multimodal data are both empty.",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  const bool supplied_tokens = prompt_tokens.has_value();
  std::vector<int32_t> tokens =
      supplied_tokens ? std::move(*prompt_tokens) : std::vector<int32_t>{};
  if (supplied_tokens && model_args_->vocab_size() > 0) {
    if (const auto invalid =
            util::find_out_of_vocab_token(tokens, model_args_->vocab_size())) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Prompt token id out of vocabulary range: " +
                              std::to_string(*invalid),
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return nullptr;
    }
  }
  const bool processed =
      supplied_tokens ? processor_->process_prompt_tokens(mm_data, tokens)
                      : processor_->process_prompt(
                            prompt, mm_data, tokens, sp.add_special_tokens);
  if (!processed) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to process prompt.",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }
  if (tokens.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt tokens are empty.",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  rate_limit_guard.dismiss();
  return build_request(std::move(prompt),
                       std::move(tokens),
                       std::move(mm_data),
                       sp,
                       call,
                       std::move(callback),
                       generation_mode,
                       arrival_time);
}

std::shared_ptr<Request> VLMRequestFactory::build_request(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    std::optional<Call*> call,
    OutputCallback callback,
    std::optional<ChatTemplateGenerationMode> generation_mode,
    std::chrono::steady_clock::time_point arrival_time) {
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
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Prompt is too long",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  const int32_t generated_context_len =
      max_context_len - options_->num_speculative_tokens();
  const int64_t remaining = static_cast<int64_t>(generated_context_len) -
                            static_cast<int64_t>(prompt_tokens.size());
  if (remaining <= 0 || sp.min_tokens > static_cast<uint64_t>(remaining)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "The remaining model context cannot satisfy min_tokens",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }
  const uint32_t max_tokens =
      sp.max_tokens == 0
          ? static_cast<uint32_t>(remaining)
          : std::min(sp.max_tokens, static_cast<uint32_t>(remaining));
  const uint32_t effective_max_tokens =
      sp.is_sample_request
          ? std::max(max_tokens, static_cast<uint32_t>(sp.sample_slots.size()))
          : max_tokens;

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens, TODO: add image token size as well.
  size_t capacity = prompt_tokens.size() + effective_max_tokens +
                    options_->num_speculative_tokens() + /*bonus_token*/ 1;
  if (options_->enable_schedule_overlap()) {
    capacity += options_->num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param = sp.to_sampling_param(best_of);
  sampling_param.device_min_tokens = sp.min_tokens > 0;
  sampling_param.json_object =
      ServiceConfig::get_instance().enable_json_object_output() &&
      sp.response_format == ResponseFormatType::JSON_OBJECT;
  if (const auto error =
          sp.prepare_sampling_constraints(sampling_param,
                                          tokenizer_,
                                          model_args_->vocab_size(),
                                          model_args_->eos_token_id(),
                                          model_args_->stop_token_ids())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        *error,
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }
  // Model-aware complement to verify_params' 2000 cap: an oversized top-k
  // would throw inside the sampler and take the whole batch down with it.
  if (const auto error =
          sampling_param.top_logprobs_vocab_error(model_args_->vocab_size())) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        *error,
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  std::optional<StoppingChecker> stopping_checker = build_stopping_checker(
      sp, effective_max_tokens, generated_context_len, callback);
  if (!stopping_checker.has_value()) {
    return nullptr;
  }
  if (options_->task_type() != "embed" && options_->task_type() != "mm_embed" &&
      stopping_checker->check(prompt_tokens, prompt_tokens.size()) !=
          FinishReason::NONE) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Invalid Prompt",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return nullptr;
  }

  if (sp.stop.has_value()) {
    std::vector<std::vector<int32_t>> generated_sequences;
    std::vector<std::string> generated_strings;
    // Byte-level tokenizers can merge a stop string with following whitespace.
    // Keep these variants out of prompt validation, as on the LLM path.
    for (const auto& stop : *sp.stop) {
      for (const std::string_view suffix : {"\n", "\r\n", " "}) {
        std::string candidate(stop);
        candidate.append(suffix);
        std::vector<int32_t> token_ids;
        if (tokenizer_->encode(candidate, &token_ids) && !token_ids.empty() &&
            std::find(generated_sequences.begin(),
                      generated_sequences.end(),
                      token_ids) == generated_sequences.end()) {
          generated_sequences.push_back(std::move(token_ids));
          generated_strings.push_back(stop);
        }
      }
    }
    stopping_checker->set_generated_stop_sequences(
        std::move(generated_sequences), std::move(generated_strings));
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
                         batch_callback_,
                         sp.decode_address,
                         call);
  req_state.scheduler_param = sp.to_scheduler_param();
  req_state.sample_slots = sp.sample_slots;
  req_state.arrival_time = arrival_time;
  if (req_state.mm_data.has(MMType::AUDIO)) {
    std::vector<AudioMetadata> audio_metadata;
    req_state.mm_data.get_metadata(MMType::AUDIO, audio_metadata);
    for (const AudioMetadata& metadata : audio_metadata) {
      req_state.input_audio_duration_s += metadata.duration;
    }
  }
  req_state.include_stop_str_in_output = sp.include_stop_str_in_output;
  if (!sp.source_xservice_addr.empty()) {
    req_state.response_type = sp.response_type;
    req_state.response_model = sp.response_model;
    req_state.response_tools = sp.response_tools;
    req_state.response_tool_choice = sp.response_tool_choice;
    req_state.response_chat_template_kwargs = sp.response_chat_template_kwargs;
    req_state.response_stream_options = sp.response_stream_options;
  }
  if (req_state.sampling_param.json_object &&
      !apply_json_object_grammar(req_state, sp, generation_mode, callback)) {
    return nullptr;
  }

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
  }
  if (!sp.ignore_eos) {
    stop_tokens.insert(model_args_->stop_token_ids().begin(),
                       model_args_->stop_token_ids().end());
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int32_t> tmp_tokens;
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

  return StoppingChecker(max_tokens,
                         max_context_len,
                         model_args_->eos_token_id(),
                         sp.ignore_eos,
                         std::move(stop_tokens),
                         std::move(stop_sequences),
                         sp.stop.value_or(std::vector<std::string>{}),
                         sp.min_tokens);
}

bool VLMRequestFactory::apply_json_object_grammar(
    RequestState& state,
    const RequestParams& sp,
    std::optional<ChatTemplateGenerationMode> generation_mode,
    const OutputCallback& callback) {
  if (generation_mode == ChatTemplateGenerationMode::UNKNOWN) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "JSON object constraint requires a recognizable chat generation mode",
        sp.service_request_id,
        sp.source_xservice_addr);
    return false;
  }
  const bool reasoning_enabled =
      generation_mode.has_value()
          ? generation_mode == ChatTemplateGenerationMode::REASONING
          : get_enable_thinking(sp.chat_template_kwargs);
  std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
  auto& grammar =
      reasoning_enabled ? json_reasoning_grammar_ : json_object_grammar_;
  if (grammar == nullptr) {
    std::string error;
    grammar =
        JsonObjectGrammar::create_from_tokenizer(*tokenizer_,
                                                 model_args_->eos_token_id(),
                                                 model_args_->stop_token_ids(),
                                                 model_args_->vocab_size(),
                                                 reasoning_enabled,
                                                 &error);
    if (grammar == nullptr) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "Failed to initialize json_object constraint: " + error,
          sp.service_request_id,
          sp.source_xservice_addr);
      return false;
    }
  }
  state.json_object_grammar = grammar;
  state.json_reasoning_enabled = reasoning_enabled;
  return true;
}

std::shared_ptr<Request> VLMRequestFactory::create(
    std::vector<Message> messages,
    const RequestParams& sp,
    MMInput mm_input,
    OutputCallback callback,
    bool use_prompt_as_is,
    std::chrono::steady_clock::time_point arrival_time) {
  return create(std::move(messages),
                std::nullopt,
                sp,
                std::move(mm_input),
                std::nullopt,
                std::move(callback),
                use_prompt_as_is,
                arrival_time);
}

std::shared_ptr<Request> VLMRequestFactory::create(
    std::vector<Message> messages,
    std::optional<std::vector<int32_t>> prompt_tokens,
    const RequestParams& sp,
    MMInput mm_input,
    std::optional<Call*> call,
    OutputCallback callback,
    bool use_prompt_as_is,
    std::chrono::steady_clock::time_point arrival_time) {
  ScopeGuard rate_limit_guard(
      [this] { rate_limiter_->decrease_one_request(); });
  auto prepared = prepare_chat_prompt(
      std::move(messages), sp, std::move(mm_input), use_prompt_as_is, callback);
  if (!prepared.has_value()) {
    return nullptr;
  }
  rate_limit_guard.dismiss();
  return create(std::move(prepared->text),
                std::move(prompt_tokens),
                std::move(prepared->mm_data),
                sp,
                call,
                std::move(callback),
                prepared->generation_mode,
                arrival_time);
}

std::optional<VLMRequestFactory::PreparedPrompt>
VLMRequestFactory::prepare_chat_prompt(std::vector<Message> messages,
                                       const RequestParams& sp,
                                       MMInput mm_inputs,
                                       bool use_prompt_as_is,
                                       const OutputCallback& callback) {
  if (mm_inputs.empty()) {
    static MMInputTransfer mm_input_transfer;
    Timer load_timer;
    const MMErrCode code = mm_input_transfer.trans(messages, mm_inputs);
    HISTOGRAM_OBSERVE(mm_input_loading_latency_milliseconds,
                      static_cast<int64_t>(load_timer.elapsed_milliseconds()));
    if (code != MMErrCode::SUCCESS) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          MMErrToString(code),
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return std::nullopt;
    }
  }

  MMData mm_data;
  if (!mm_inputs.empty()) {
    Timer preprocess_timer;
    const bool processed = processor_->process_multimodal(mm_inputs, mm_data);
    HISTOGRAM_OBSERVE(
        mm_preprocess_latency_milliseconds,
        static_cast<int64_t>(preprocess_timer.elapsed_milliseconds()));
    if (!processed) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to process multimodal input.",
                          sp.service_request_id,
                          sp.source_xservice_addr);
      return std::nullopt;
    }
  }

  Timer timer;
  std::optional<std::string> prompt;
  std::optional<ChatTemplateGenerationMode> generation_mode;
  if (use_prompt_as_is) {
    prompt = join_message_texts(messages);
    if (!prompt.has_value()) {
      throw_prompt_as_is_error(messages);
    }
  } else {
    nlohmann::ordered_json kwargs = sp.chat_template_kwargs;
    if (model_args_->audio_token_id() != 0) {
      kwargs["use_ctc"] = ModelConfig::get_instance().use_ctc();
    }
    auto rendered =
        chat_template_->apply_with_generation_mode(messages, sp.tools, kwargs);
    if (rendered.has_value()) {
      prompt = std::move(rendered->prompt);
      generation_mode = rendered->generation_mode;
    }
  }
  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages",
                        sp.service_request_id,
                        sp.source_xservice_addr);
    return std::nullopt;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());
  return PreparedPrompt{
      std::move(*prompt), std::move(mm_data), generation_mode};
}

Status VLMRequestFactory::count_chat_tokens(std::vector<Message> messages,
                                            const RequestParams& sp,
                                            MMInput mm_input,
                                            int32_t& token_count) {
  Status error;
  const OutputCallback callback = [&error](const RequestOutput& output) {
    error = output.status.value();
    return true;
  };
  auto prepared = prepare_chat_prompt(std::move(messages),
                                      sp,
                                      std::move(mm_input),
                                      /*use_prompt_as_is=*/false,
                                      callback);
  if (!prepared.has_value()) {
    return error;
  }
  std::vector<int32_t> tokens;
  if (!processor_->process_prompt(
          prepared->text, prepared->mm_data, tokens, sp.add_special_tokens)) {
    return Status(StatusCode::INVALID_ARGUMENT, "Failed to process prompt");
  }
  token_count = static_cast<int32_t>(tokens.size());
  return Status();
}

}  // namespace xllm

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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/message.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "framework/chat_template/chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/request_state.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/json_object_grammar.h"

namespace xllm {

class Call;
class Tokenizer;

// Factory that assembles a domain Request aggregate from raw user input
// (a text prompt or chat messages) plus RequestParams. It encapsulates prompt
// encoding, token/length validation, sampling/scheduler/stopping parameter
// assembly, and optional JSON-object grammar construction.
//
// The factory holds non-owning references to model configuration owned by
// LLMMaster (tokenizer, chat template, model args, options, rate limiter); the
// owner must outlive the factory. Only the JSON-object grammar cache is owned
// here since it is built lazily and used solely during request creation.
class LLMRequestFactory final {
 public:
  // Turns a batch of request outputs into an xllm-service RPC response.
  // Injected by LLMMaster so the factory can install a batch callback when
  // service routing is enabled, without depending on LLMMaster directly.
  using RpcResponseHandler =
      std::function<std::vector<bool>(const std::vector<RequestOutput>&)>;

  LLMRequestFactory(const Tokenizer* tokenizer,
                    const ChatTemplate* chat_template,
                    const ModelArgs* model_args,
                    const Options* options,
                    RateLimiter* rate_limiter,
                    std::string task_type,
                    RpcResponseHandler rpc_response_handler);

  // completion / encode: prompt carries text and/or pre-tokenized tokens.
  std::shared_ptr<Request> create(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback,
      std::optional<ChatTemplateGenerationMode> generation_mode = std::nullopt);

  // chat: renders messages through the chat template, then delegates to the
  // text-prompt overload above.
  std::shared_ptr<Request> create(const std::vector<Message>& messages,
                                  std::optional<std::vector<int>> prompt_tokens,
                                  const RequestParams& sp,
                                  std::optional<Call*> call,
                                  OutputCallback callback);

 private:
  // Encodes the prompt (or accepts pre-tokenized input) and validates it
  // against the vocabulary range and the context-length limit. Returns the
  // token ids, or std::nullopt after firing an error callback.
  std::optional<std::vector<int>> encode_and_validate_prompt(
      const std::string& prompt,
      std::optional<std::vector<int>> prompt_tokens,
      int32_t max_context_len,
      const RequestParams& sp,
      const OutputCallback& callback);

  RequestSamplingParam build_sampling_param(const RequestParams& sp,
                                            size_t best_of) const;

  // Builds the stopping checker, encoding any stop sequences. Returns
  // std::nullopt after firing an error callback when a stop sequence fails to
  // encode.
  std::optional<StoppingChecker> build_stopping_checker(
      const RequestParams& sp,
      uint32_t effective_max_tokens,
      int32_t max_context_len,
      const OutputCallback& callback);

  // Rejects a prompt that already ends in a stop condition (skipped for
  // embedding tasks). Returns false after firing an error callback.
  bool validate_prompt_not_finished(const StoppingChecker& stopping_checker,
                                    const std::vector<int>& prompt_tokens,
                                    const RequestParams& sp,
                                    const OutputCallback& callback) const;

  // Attaches the JSON-object grammar to req_state when requested. Returns
  // false after firing an error callback on failure.
  bool apply_json_object_grammar(
      RequestState& req_state,
      const RequestParams& sp,
      std::optional<ChatTemplateGenerationMode> generation_mode,
      const OutputCallback& callback);

  std::shared_ptr<const JsonObjectGrammar> get_json_object_grammar(
      bool reasoning_enabled,
      std::string* error);

  const Tokenizer* tokenizer_ = nullptr;
  const ChatTemplate* chat_template_ = nullptr;
  const ModelArgs* model_args_ = nullptr;
  const Options* options_ = nullptr;
  RateLimiter* rate_limiter_ = nullptr;
  std::string task_type_;
  RpcResponseHandler rpc_response_handler_;

  std::mutex json_object_grammar_mutex_;
  std::shared_ptr<const JsonObjectGrammar> json_object_grammar_;
  std::shared_ptr<const JsonObjectGrammar> json_reasoning_grammar_;
};

}  // namespace xllm

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

#include <chrono>
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
#include "core/framework/multimodal/mm_data.h"
#include "framework/chat_template/chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/request_state.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {

class Tokenizer;
class Call;
class JsonObjectGrammar;

// Factory that assembles a domain Request aggregate for the VLM path from raw
// user input (a text prompt with multimodal data, or chat messages plus a
// multimodal payload) and RequestParams. It encapsulates multimodal
// preprocessing, prompt tokenization via the processor, chat-template
// rendering, length validation, and sampling/stopping parameter assembly.
//
// The factory holds non-owning references to model configuration owned by
// VLMMaster (processor, chat template, tokenizer, model args, options, rate
// limiter); the owner must outlive the factory.
class VLMRequestFactory final {
 public:
  using RpcResponseHandler =
      std::function<std::vector<bool>(const std::vector<RequestOutput>&)>;

  VLMRequestFactory(MultimodalProcessorBase* processor,
                    ChatTemplate* chat_template,
                    const Tokenizer* tokenizer,
                    const ModelArgs* model_args,
                    const Options* options,
                    RateLimiter* rate_limiter,
                    RpcResponseHandler rpc_response_handler = {});

  // completion: a text prompt plus already-decoded multimodal data.
  std::shared_ptr<Request> create(
      std::string prompt,
      MMData mm_data,
      const RequestParams& sp,
      OutputCallback callback,
      std::chrono::steady_clock::time_point arrival_time =
          std::chrono::steady_clock::now());

  std::shared_ptr<Request> create(
      std::string prompt,
      std::optional<std::vector<int32_t>> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback,
      std::optional<ChatTemplateGenerationMode> generation_mode = std::nullopt,
      std::chrono::steady_clock::time_point arrival_time =
          std::chrono::steady_clock::now());

  // chat: decodes the multimodal payload and renders messages into a prompt,
  // then delegates to the completion overload above.
  std::shared_ptr<Request> create(
      std::vector<Message> messages,
      const RequestParams& sp,
      MMInput mm_input,
      OutputCallback callback,
      bool use_prompt_as_is = false,
      std::chrono::steady_clock::time_point arrival_time =
          std::chrono::steady_clock::now());

  std::shared_ptr<Request> create(
      std::vector<Message> messages,
      std::optional<std::vector<int32_t>> prompt_tokens,
      const RequestParams& sp,
      MMInput mm_input,
      std::optional<Call*> call,
      OutputCallback callback,
      bool use_prompt_as_is = false,
      std::chrono::steady_clock::time_point arrival_time =
          std::chrono::steady_clock::now());

  // Includes processor-expanded image/video/audio placeholders, without
  // constructing a Request or taking a generation rate-limit slot.
  Status count_chat_tokens(std::vector<Message> messages,
                           const RequestParams& sp,
                           MMInput mm_input,
                           int32_t& token_count);

 private:
  struct PreparedPrompt {
    std::string text;
    MMData mm_data;
    std::optional<ChatTemplateGenerationMode> generation_mode;
  };

  std::optional<PreparedPrompt> prepare_chat_prompt(
      std::vector<Message> messages,
      const RequestParams& sp,
      MMInput mm_input,
      bool use_prompt_as_is,
      const OutputCallback& callback);

  std::shared_ptr<Request> build_request(
      std::string prompt,
      std::vector<int32_t> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      std::optional<Call*> call,
      OutputCallback callback,
      std::optional<ChatTemplateGenerationMode> generation_mode,
      std::chrono::steady_clock::time_point arrival_time);

  // Builds the stopping checker, encoding any stop sequences. Returns
  // std::nullopt after firing an error callback when a stop sequence fails to
  // encode.
  std::optional<StoppingChecker> build_stopping_checker(
      const RequestParams& sp,
      uint32_t max_tokens,
      int32_t max_context_len,
      const OutputCallback& callback);

  bool apply_json_object_grammar(
      RequestState& state,
      const RequestParams& sp,
      std::optional<ChatTemplateGenerationMode> generation_mode,
      const OutputCallback& callback);

  MultimodalProcessorBase* processor_ = nullptr;
  ChatTemplate* chat_template_ = nullptr;
  const Tokenizer* tokenizer_ = nullptr;
  const ModelArgs* model_args_ = nullptr;
  const Options* options_ = nullptr;
  RateLimiter* rate_limiter_ = nullptr;
  OutputsFunc batch_callback_;
  std::mutex json_object_grammar_mutex_;
  std::shared_ptr<const JsonObjectGrammar> json_object_grammar_;
  std::shared_ptr<const JsonObjectGrammar> json_reasoning_grammar_;
};

}  // namespace xllm

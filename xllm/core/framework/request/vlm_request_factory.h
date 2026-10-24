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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/message.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/chat_template/jinja_chat_template.h"
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
  VLMRequestFactory(MultimodalProcessorBase* processor,
                    JinjaChatTemplate* chat_template,
                    const Tokenizer* tokenizer,
                    const ModelArgs* model_args,
                    const Options* options,
                    RateLimiter* rate_limiter);

  // completion: a text prompt plus already-decoded multimodal data.
  std::shared_ptr<Request> create(std::string prompt,
                                  MMData mm_data,
                                  const RequestParams& sp,
                                  OutputCallback callback);

  // chat: decodes the multimodal payload and renders messages into a prompt,
  // then delegates to the completion overload above.
  std::shared_ptr<Request> create(std::vector<Message> messages,
                                  const RequestParams& sp,
                                  std::string payload,
                                  OutputCallback callback,
                                  bool use_prompt_as_is = false);

 private:
  std::shared_ptr<Request> build_request(std::string prompt,
                                         std::vector<int32_t> prompt_tokens,
                                         MMData mm_data,
                                         const RequestParams& sp,
                                         OutputCallback callback);

  // Builds the stopping checker, encoding any stop sequences. Returns
  // std::nullopt after firing an error callback when a stop sequence fails to
  // encode.
  std::optional<StoppingChecker> build_stopping_checker(
      const RequestParams& sp,
      uint32_t max_tokens,
      int32_t max_context_len,
      const OutputCallback& callback);

  MultimodalProcessorBase* processor_ = nullptr;
  JinjaChatTemplate* chat_template_ = nullptr;
  const Tokenizer* tokenizer_ = nullptr;
  const ModelArgs* model_args_ = nullptr;
  const Options* options_ = nullptr;
  RateLimiter* rate_limiter_ = nullptr;
};

}  // namespace xllm

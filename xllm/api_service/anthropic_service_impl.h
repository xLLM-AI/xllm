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

#include <type_traits>

#include "anthropic.pb.h"
#include "api_service/api_service_impl.h"
#include "api_service/stream_call.h"

namespace xllm {

// Specialize is_stream_call for AnthropicCall to recognize it as a stream call
template <>
struct is_stream_call<AnthropicCall> : std::true_type {};

class AnthropicServiceImpl final : public APIServiceImpl<AnthropicCall> {
 public:
  AnthropicServiceImpl(LLMMaster* master,
                       const std::vector<std::string>& models);

  void process_async_impl(std::shared_ptr<AnthropicCall> call) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(AnthropicServiceImpl);

  LLMMaster* master_ = nullptr;
  const std::string tool_call_parser_format_;
  const std::string reasoning_parser_format_;
};

}  // namespace xllm

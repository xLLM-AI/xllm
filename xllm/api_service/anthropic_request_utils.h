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

#pragma once

#include <string>
#include <vector>

#include "anthropic.pb.h"
#include "core/common/message.h"
#include "core/common/types.h"

namespace xllm::api_service {

Status parse_anthropic_request(const std::string& json,
                               bool count_tokens,
                               proto::AnthropicMessagesRequest& request);

Status validate_anthropic_request(
    const proto::AnthropicMessagesRequest& request,
    bool count_tokens = false);

Status validate_anthropic_backend(
    const proto::AnthropicMessagesRequest& request,
    bool count_tokens = false);

std::vector<Message> build_anthropic_messages(
    const proto::AnthropicMessagesRequest& request);

}  // namespace xllm::api_service

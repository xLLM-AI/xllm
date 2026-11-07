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

#include <memory>
#include <string>

#include "core/framework/request/request_output.h"

namespace xllm::proto {
class DisaggRequest;
}  // namespace xllm::proto

namespace xllm {

// Per-request parser state, shared by the request and its queued outputs.
// XServiceClient selects typed RPC frames or encoded HTTP output.
std::shared_ptr<ServiceOutputFormatter> make_decode_output_formatter(
    const proto::DisaggRequest& request,
    const std::string& tool_parser_format,
    const std::string& reasoning_parser_format,
    std::string system_fingerprint);

}  // namespace xllm
